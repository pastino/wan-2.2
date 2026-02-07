# Wan2.2 Video Generation RunPod Serverless Dockerfile
# 빌드: docker build -t wan2.2-runpod -f runpod/Dockerfile .
# (Wan2.2 루트 디렉토리에서 실행)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# GPU 아키텍처 지정 (CI에서 GPU 없이 CUDA 확장 컴파일 시 필요)
# A100=8.0, H100=9.0
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"

# pip 업그레이드
RUN pip install --upgrade pip

# Wan2.2 기본 의존성 설치 (flash_attn은 torch 의존성 때문에 별도 설치)
COPY requirements.txt /app/requirements.txt
RUN grep -v "flash_attn" /app/requirements.txt > /app/requirements_no_flash.txt && \
    pip install -r /app/requirements_no_flash.txt

# S2V 추가 의존성
COPY requirements_s2v.txt /app/requirements_s2v.txt
RUN pip install -r /app/requirements_s2v.txt

# Animate 추가 의존성 (SAM-2는 editable 설치 불가 → 별도 처리)
COPY requirements_animate.txt /app/requirements_animate.txt
RUN grep -v "sam2" /app/requirements_animate.txt > /app/requirements_animate_no_sam.txt && \
    pip install -r /app/requirements_animate_no_sam.txt && \
    pip install git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266

# flash-attn 별도 설치 (빌드 격리 비활성화 필요)
RUN pip install flash-attn --no-build-isolation

# RunPod SDK, AWS SDK, HuggingFace 설치
RUN pip install runpod boto3 requests huggingface_hub

# Wan2.2 소스 복사
COPY wan/ /app/wan/
COPY generate.py /app/generate.py
COPY runpod/handler.py /app/handler.py

# 모델은 런타임에 다운로드 (RunPod Network Volume 캐시)
# HF_TOKEN 환경변수로 HuggingFace 인증

CMD ["python", "-u", "/app/handler.py"]
