"""
Wan2.2 Video Generation RunPod Serverless Handler

지원 태스크: t2v, i2v, ti2v, s2v, animate
모델: H100 80GB 기준, 14B/5B 모델 지원

환경변수:
- HF_TOKEN: HuggingFace 모델 다운로드 토큰
- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: S3 업로드
- AWS_REGION: S3 리전 (기본: ap-northeast-2)
- AWS_S3_BUCKET: 기본 S3 버킷
- MODEL_DIR: 모델 체크포인트 경로 (기본: /runpod-volume/wan2.2-models)
"""

import gc
import logging
import os
import random
import sys
import tempfile
import time
import uuid
from datetime import datetime
from io import BytesIO

import boto3
import requests
import runpod
import torch
from PIL import Image

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)])

# HuggingFace 토큰 설정
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_HUB_TOKEN"] = hf_token
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        logging.info("HuggingFace 로그인 완료")
    except Exception as e:
        logging.warning(f"HuggingFace 로그인 실패: {e}")

# Wan2.2 임포트
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video

# 모델 경로
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/wan2.2-models")

# task → WAN_CONFIGS 키 매핑
TASK_CONFIG_MAP = {
    "t2v": "t2v-A14B",
    "i2v": "i2v-A14B",
    "ti2v": "ti2v-5B",
    "s2v": "s2v-14B",
    "animate": "animate-14B",
}

# task → 모델 디렉토리 매핑
TASK_MODEL_DIR_MAP = {
    "t2v": "Wan2.2-T2V-A14B",
    "i2v": "Wan2.2-I2V-A14B",
    "ti2v": "Wan2.2-TI2V-5B",
    "s2v": "Wan2.2-S2V-14B",
    "animate": "Wan2.2-Animate-14B",
}

# 전역 파이프라인 캐시 (콜드 스타트 최적화)
current_pipeline = None
current_task = None

# S3 클라이언트 (싱글톤)
s3_client = None


def get_s3_client():
    """S3 클라이언트 초기화"""
    global s3_client
    if s3_client is None:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "ap-northeast-2"),
        )
    return s3_client


def download_file(url: str, suffix: str = "") -> str:
    """URL에서 파일 다운로드 → 임시 파일 반환"""
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    if not suffix:
        # URL에서 확장자 추출
        path = url.split("?")[0]
        if "." in path.split("/")[-1]:
            suffix = "." + path.split("/")[-1].split(".")[-1]
        else:
            suffix = ".tmp"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(response.content)
    tmp.close()
    logging.info(f"파일 다운로드 완료: {url[:80]}... → {tmp.name}")
    return tmp.name


def upload_to_s3(file_path: str, bucket: str, folder: str = "generated-videos") -> str:
    """MP4 파일을 S3에 업로드하고 URL 반환"""
    client = get_s3_client()

    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{folder}/{today}/{uuid.uuid4().hex}.mp4"

    client.upload_file(
        file_path,
        bucket,
        filename,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    region = os.environ.get("AWS_REGION", "ap-northeast-2")
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{filename}"
    logging.info(f"S3 업로드 완료: {url}")
    return url


def unload_pipeline():
    """현재 로드된 파이프라인 해제 → GPU 메모리 확보"""
    global current_pipeline, current_task
    if current_pipeline is not None:
        logging.info(f"파이프라인 해제 중: {current_task}")
        del current_pipeline
        current_pipeline = None
        current_task = None
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"GPU 메모리 해제 완료 (free: {torch.cuda.mem_get_info()[0] / 1e9:.1f}GB)")


def load_pipeline(task: str):
    """
    요청된 task에 맞는 파이프라인 로드.
    이미 같은 task의 파이프라인이 로드되어 있으면 재사용.
    다른 task면 기존 해제 후 새로 로드.
    """
    global current_pipeline, current_task

    if current_task == task and current_pipeline is not None:
        logging.info(f"파이프라인 재사용: {task}")
        return current_pipeline

    # 다른 모델이 로드되어 있으면 해제
    if current_pipeline is not None:
        unload_pipeline()

    config_key = TASK_CONFIG_MAP[task]
    cfg = WAN_CONFIGS[config_key]
    ckpt_dir = os.path.join(MODEL_DIR, TASK_MODEL_DIR_MAP[task])

    logging.info(f"파이프라인 로드 시작: {task} (ckpt: {ckpt_dir})")
    start = time.time()

    if task == "t2v":
        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
        )
    elif task == "i2v":
        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
        )
    elif task == "ti2v":
        pipeline = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
        )
    elif task == "s2v":
        pipeline = wan.WanS2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
        )
    elif task == "animate":
        pipeline = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
        )
    else:
        raise ValueError(f"지원하지 않는 task: {task}")

    elapsed = time.time() - start
    logging.info(f"파이프라인 로드 완료: {task} ({elapsed:.1f}s)")

    current_pipeline = pipeline
    current_task = task
    return pipeline


def generate_t2v(pipeline, job_input: dict, cfg) -> torch.Tensor:
    """텍스트 → 영상 생성"""
    prompt = job_input["prompt"]
    size = job_input.get("size", "1280*720")
    frame_num = job_input.get("frame_num", cfg.frame_num)
    seed = job_input.get("seed", -1)
    sample_steps = job_input.get("sample_steps", cfg.sample_steps)
    sample_guide_scale = job_input.get("sample_guide_scale", cfg.sample_guide_scale)
    sample_shift = job_input.get("sample_shift", cfg.sample_shift)

    if seed < 0:
        seed = random.randint(0, sys.maxsize)

    logging.info(f"T2V 생성: size={size}, frames={frame_num}, steps={sample_steps}, seed={seed}")

    video = pipeline.generate(
        prompt,
        size=SIZE_CONFIGS[size],
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver="unipc",
        sampling_steps=sample_steps,
        guide_scale=sample_guide_scale,
        seed=seed,
        offload_model=True,
    )
    return video


def generate_i2v(pipeline, job_input: dict, cfg) -> torch.Tensor:
    """이미지 → 영상 생성"""
    prompt = job_input["prompt"]
    image_path = job_input["_image_path"]
    size = job_input.get("size", "1280*720")
    frame_num = job_input.get("frame_num", cfg.frame_num)
    seed = job_input.get("seed", -1)
    sample_steps = job_input.get("sample_steps", cfg.sample_steps)
    sample_guide_scale = job_input.get("sample_guide_scale", cfg.sample_guide_scale)
    sample_shift = job_input.get("sample_shift", cfg.sample_shift)

    if seed < 0:
        seed = random.randint(0, sys.maxsize)

    img = Image.open(image_path).convert("RGB")

    logging.info(f"I2V 생성: size={size}, frames={frame_num}, steps={sample_steps}, seed={seed}")

    video = pipeline.generate(
        prompt,
        img,
        max_area=MAX_AREA_CONFIGS[size],
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver="unipc",
        sampling_steps=sample_steps,
        guide_scale=sample_guide_scale,
        seed=seed,
        offload_model=True,
    )
    return video


def generate_ti2v(pipeline, job_input: dict, cfg) -> torch.Tensor:
    """텍스트+이미지 → 영상 생성 (5B 경량)"""
    prompt = job_input["prompt"]
    image_path = job_input["_image_path"]
    size = job_input.get("size", "1280*704")
    frame_num = job_input.get("frame_num", cfg.frame_num)
    seed = job_input.get("seed", -1)
    sample_steps = job_input.get("sample_steps", cfg.sample_steps)
    sample_guide_scale = job_input.get("sample_guide_scale", cfg.sample_guide_scale)
    sample_shift = job_input.get("sample_shift", cfg.sample_shift)

    if seed < 0:
        seed = random.randint(0, sys.maxsize)

    img = Image.open(image_path).convert("RGB")

    logging.info(f"TI2V 생성: size={size}, frames={frame_num}, steps={sample_steps}, seed={seed}")

    video = pipeline.generate(
        prompt,
        img=img,
        size=SIZE_CONFIGS[size],
        max_area=MAX_AREA_CONFIGS[size],
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver="unipc",
        sampling_steps=sample_steps,
        guide_scale=sample_guide_scale,
        seed=seed,
        offload_model=True,
    )
    return video


def generate_s2v(pipeline, job_input: dict, cfg) -> tuple:
    """음성 → 영상 생성. (video, audio_path) 반환"""
    prompt = job_input["prompt"]
    image_path = job_input["_image_path"]
    audio_path = job_input["_audio_path"]
    size = job_input.get("size", "1024*704")
    seed = job_input.get("seed", -1)
    sample_steps = job_input.get("sample_steps", cfg.sample_steps)
    sample_guide_scale = job_input.get("sample_guide_scale", cfg.sample_guide_scale)
    sample_shift = job_input.get("sample_shift", cfg.sample_shift)
    infer_frames = job_input.get("infer_frames", 80)
    num_clip = job_input.get("num_clip", None)

    if seed < 0:
        seed = random.randint(0, sys.maxsize)

    logging.info(f"S2V 생성: size={size}, steps={sample_steps}, seed={seed}")

    video = pipeline.generate(
        input_prompt=prompt,
        ref_image_path=image_path,
        audio_path=audio_path,
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        num_repeat=num_clip,
        max_area=MAX_AREA_CONFIGS[size],
        infer_frames=infer_frames,
        shift=sample_shift,
        sample_solver="unipc",
        sampling_steps=sample_steps,
        guide_scale=sample_guide_scale,
        seed=seed,
        offload_model=True,
    )
    return video, audio_path


def generate_animate(pipeline, job_input: dict, cfg) -> torch.Tensor:
    """캐릭터 애니메이션 생성"""
    # animate는 src_root_path에 이미지+비디오가 정리되어 있어야 함
    src_root_path = job_input["_src_root_path"]
    replace_flag = job_input.get("replace_flag", False)
    frame_num = job_input.get("frame_num", cfg.frame_num)
    # generate.py CLI 기본값 77 (temporal guidance 프레임 수)
    refert_num = job_input.get("refert_num", 77)
    seed = job_input.get("seed", -1)
    sample_steps = job_input.get("sample_steps", cfg.sample_steps)
    sample_guide_scale = job_input.get("sample_guide_scale", cfg.sample_guide_scale)
    sample_shift = job_input.get("sample_shift", cfg.sample_shift)

    if seed < 0:
        seed = random.randint(0, sys.maxsize)

    logging.info(f"Animate 생성: frames={frame_num}, refert_num={refert_num}, steps={sample_steps}, seed={seed}")

    video = pipeline.generate(
        src_root_path=src_root_path,
        replace_flag=replace_flag,
        refert_num=refert_num,
        clip_len=frame_num,
        shift=sample_shift,
        sample_solver="unipc",
        sampling_steps=sample_steps,
        guide_scale=sample_guide_scale,
        seed=seed,
        offload_model=True,
    )
    return video


def prepare_animate_source(image_path: str, video_path: str) -> str:
    """
    Animate task용 소스 디렉토리 준비.
    이미지와 비디오를 임시 디렉토리에 정리.
    """
    src_dir = tempfile.mkdtemp(prefix="wan_animate_")
    # animate 파이프라인이 요구하는 디렉토리 구조에 맞게 파일 배치
    import shutil
    shutil.copy2(image_path, os.path.join(src_dir, "reference.jpg"))
    shutil.copy2(video_path, os.path.join(src_dir, "driving.mp4"))
    return src_dir


def handler(job):
    """RunPod Serverless Handler"""
    try:
        job_input = job["input"]
        task = job_input.get("task")

        if not task:
            return {"error": "task 파라미터가 필요합니다. (t2v, i2v, ti2v, s2v, animate)"}

        if task not in TASK_CONFIG_MAP:
            return {"error": f"지원하지 않는 task: {task}. 가능: {list(TASK_CONFIG_MAP.keys())}"}

        # S3 설정
        bucket = job_input.get("s3_bucket") or os.environ.get("AWS_S3_BUCKET")
        folder = job_input.get("s3_folder", "generated-videos")
        if not bucket:
            return {"error": "s3_bucket 파라미터 또는 AWS_S3_BUCKET 환경변수가 필요합니다."}

        config_key = TASK_CONFIG_MAP[task]
        cfg = WAN_CONFIGS[config_key]

        # 리소스 다운로드 (임시 파일 추적)
        temp_files = []
        temp_dirs = []

        try:
            # 이미지 다운로드
            if job_input.get("image_url"):
                image_path = download_file(job_input["image_url"])
                job_input["_image_path"] = image_path
                temp_files.append(image_path)

            # 오디오 다운로드
            if job_input.get("audio_url"):
                audio_path = download_file(job_input["audio_url"])
                job_input["_audio_path"] = audio_path
                temp_files.append(audio_path)

            # 비디오 다운로드 (animate용)
            if job_input.get("video_url"):
                video_path = download_file(job_input["video_url"])
                job_input["_video_path"] = video_path
                temp_files.append(video_path)

            # 입력 검증
            if task in ("i2v", "ti2v", "s2v") and "_image_path" not in job_input:
                return {"error": f"{task} task에는 image_url이 필요합니다."}
            if task == "s2v" and "_audio_path" not in job_input:
                return {"error": "s2v task에는 audio_url이 필요합니다."}
            if task == "animate" and ("_image_path" not in job_input or "_video_path" not in job_input):
                return {"error": "animate task에는 image_url과 video_url이 필요합니다."}
            if task in ("t2v", "i2v", "ti2v", "s2v") and not job_input.get("prompt"):
                return {"error": f"{task} task에는 prompt가 필요합니다."}

            # Animate 소스 디렉토리 준비
            if task == "animate":
                src_dir = prepare_animate_source(
                    job_input["_image_path"], job_input["_video_path"]
                )
                job_input["_src_root_path"] = src_dir
                temp_dirs.append(src_dir)

            # 파이프라인 로드
            pipeline = load_pipeline(task)

            # 영상 생성
            gen_start = time.time()

            if task == "t2v":
                video = generate_t2v(pipeline, job_input, cfg)
                audio_path_for_merge = None
            elif task == "i2v":
                video = generate_i2v(pipeline, job_input, cfg)
                audio_path_for_merge = None
            elif task == "ti2v":
                video = generate_ti2v(pipeline, job_input, cfg)
                audio_path_for_merge = None
            elif task == "s2v":
                video, audio_path_for_merge = generate_s2v(pipeline, job_input, cfg)
            elif task == "animate":
                video = generate_animate(pipeline, job_input, cfg)
                audio_path_for_merge = None

            gen_elapsed = time.time() - gen_start
            logging.info(f"영상 생성 완료: {gen_elapsed:.1f}s")

            # MP4 저장
            tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_video.close()
            temp_files.append(tmp_video.name)

            save_video(
                tensor=video[None],
                save_file=tmp_video.name,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

            # S2V: 오디오 병합
            if audio_path_for_merge:
                from wan.utils.utils import merge_video_audio
                merge_video_audio(video_path=tmp_video.name, audio_path=audio_path_for_merge)

            # video 텐서 해제
            del video
            torch.cuda.empty_cache()

            # S3 업로드
            video_url = upload_to_s3(tmp_video.name, bucket, folder)

            # 프레임 수 / FPS로 영상 길이 계산
            frame_num = job_input.get("frame_num", cfg.frame_num)
            fps = cfg.sample_fps
            duration = frame_num / fps if fps > 0 else 0

            result = {
                "video_url": video_url,
                "task": task,
                "prompt": job_input.get("prompt", ""),
                "size": job_input.get("size", ""),
                "duration_seconds": round(duration, 1),
                "generation_time_seconds": round(gen_elapsed, 1),
            }

            logging.info(f"완료: {result}")
            return result

        finally:
            # 임시 파일/디렉토리 정리
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)
            for d in temp_dirs:
                import shutil
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)

    except Exception as e:
        logging.error(f"Handler 에러: {e}", exc_info=True)
        return {"error": str(e)}


# RunPod Serverless 시작
runpod.serverless.start({"handler": handler})
