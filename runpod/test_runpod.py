"""
Wan2.2 RunPod Serverless 엔드포인트 테스트 스크립트

사용법:
    # 환경변수 설정
    export RUNPOD_API_KEY="your_api_key"
    export RUNPOD_ENDPOINT_ID="your_endpoint_id"

    # 전체 테스트
    python test_runpod.py

    # 특정 task만 테스트
    python test_runpod.py --task t2v
    python test_runpod.py --task i2v --image_url "https://..."
"""

import argparse
import json
import os
import sys
import time

import requests

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def submit_job(payload: dict) -> str:
    """작업 제출 → job_id 반환"""
    url = f"{BASE_URL}/run"
    response = requests.post(url, json=payload, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    job_id = data.get("id")
    print(f"작업 제출 완료: job_id={job_id}")
    return job_id


def poll_status(job_id: str, interval: int = 5, timeout: int = 600) -> dict:
    """작업 상태 폴링 → 완료 시 결과 반환"""
    url = f"{BASE_URL}/status/{job_id}"
    start = time.time()

    while time.time() - start < timeout:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")

        if status == "COMPLETED":
            print(f"완료! ({time.time() - start:.0f}s)")
            return data.get("output", {})
        elif status == "FAILED":
            print(f"실패: {data.get('error', 'unknown')}")
            return data
        else:
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] 상태: {status}")
            time.sleep(interval)

    print(f"타임아웃 ({timeout}s)")
    return {"error": "timeout"}


def run_job(payload: dict) -> dict:
    """작업 제출 + 폴링 → 결과 반환"""
    job_id = submit_job(payload)
    return poll_status(job_id)


def test_t2v():
    """T2V (텍스트 → 영상) 테스트"""
    print("\n=== T2V 테스트 ===")
    payload = {
        "input": {
            "task": "t2v",
            "prompt": "A cat surfing on a wave in the ocean, cinematic lighting, 4K quality",
            "size": "1280*720",
            "frame_num": 81,
            "seed": 42,
            "sample_steps": 40,
            "s3_bucket": "life-vision-dev",
            "s3_folder": "generated-videos",
        }
    }
    result = run_job(payload)
    print(f"결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
    return result


def test_i2v(image_url: str = None):
    """I2V (이미지 → 영상) 테스트"""
    print("\n=== I2V 테스트 ===")
    if not image_url:
        print("image_url이 필요합니다. --image_url 옵션으로 지정하세요.")
        return None

    payload = {
        "input": {
            "task": "i2v",
            "prompt": "The cat slowly turns its head and looks at the camera with a relaxed expression",
            "image_url": image_url,
            "size": "1280*720",
            "frame_num": 81,
            "s3_bucket": "life-vision-dev",
            "s3_folder": "generated-videos",
        }
    }
    result = run_job(payload)
    print(f"결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
    return result


def test_ti2v(image_url: str = None):
    """TI2V (텍스트+이미지 → 영상) 테스트"""
    print("\n=== TI2V 테스트 ===")
    if not image_url:
        print("image_url이 필요합니다. --image_url 옵션으로 지정하세요.")
        return None

    payload = {
        "input": {
            "task": "ti2v",
            "prompt": "Two cats fighting intensely on a spotlighted stage",
            "image_url": image_url,
            "size": "1280*704",
            "s3_bucket": "life-vision-dev",
            "s3_folder": "generated-videos",
        }
    }
    result = run_job(payload)
    print(f"결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
    return result


def test_s2v(image_url: str = None, audio_url: str = None):
    """S2V (음성 → 영상) 테스트"""
    print("\n=== S2V 테스트 ===")
    if not image_url or not audio_url:
        print("image_url과 audio_url이 필요합니다.")
        return None

    payload = {
        "input": {
            "task": "s2v",
            "prompt": "A person speaking naturally with subtle facial expressions",
            "image_url": image_url,
            "audio_url": audio_url,
            "size": "1024*704",
            "s3_bucket": "life-vision-dev",
            "s3_folder": "generated-videos",
        }
    }
    result = run_job(payload)
    print(f"결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
    return result


def test_animate(image_url: str = None, video_url: str = None):
    """Animate (캐릭터 애니메이션) 테스트"""
    print("\n=== Animate 테스트 ===")
    if not image_url or not video_url:
        print("image_url과 video_url이 필요합니다.")
        return None

    payload = {
        "input": {
            "task": "animate",
            "image_url": image_url,
            "video_url": video_url,
            "replace_flag": False,
            "s3_bucket": "life-vision-dev",
            "s3_folder": "generated-videos",
        }
    }
    result = run_job(payload)
    print(f"결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
    return result


if __name__ == "__main__":
    if not API_KEY or not ENDPOINT_ID:
        print("환경변수를 설정하세요:")
        print("  export RUNPOD_API_KEY='your_api_key'")
        print("  export RUNPOD_ENDPOINT_ID='your_endpoint_id'")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Wan2.2 RunPod 엔드포인트 테스트")
    parser.add_argument("--task", type=str, default="t2v",
                        choices=["t2v", "i2v", "ti2v", "s2v", "animate", "all"],
                        help="테스트할 task (기본: t2v)")
    parser.add_argument("--image_url", type=str, default=None, help="입력 이미지 URL")
    parser.add_argument("--audio_url", type=str, default=None, help="입력 오디오 URL")
    parser.add_argument("--video_url", type=str, default=None, help="입력 비디오 URL (animate용)")
    args = parser.parse_args()

    if args.task == "t2v" or args.task == "all":
        test_t2v()
    if args.task == "i2v" or args.task == "all":
        test_i2v(args.image_url)
    if args.task == "ti2v" or args.task == "all":
        test_ti2v(args.image_url)
    if args.task == "s2v" or args.task == "all":
        test_s2v(args.image_url, args.audio_url)
    if args.task == "animate" or args.task == "all":
        test_animate(args.image_url, args.video_url)
