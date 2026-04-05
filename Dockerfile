# 베이스 이미지로 용량이 작고 가벼운 python-slim 사용
FROM python:3.12-slim

# 환경변수 설정 (파이썬 바이트코드(pyc) 생성 방지 및 버퍼링 끔)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 라우팅 기본 작업 디렉토리 지정
WORKDIR /app

# OpenCV 및 YOLO 동작에 필요한 시스템 라이브러리 설치 (설치 후 캐시를 비워 이미지 크기 최적화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 패키지 설치용 파일 복사
COPY requirements.txt .

# pip 캐시를 사용하지 않아 이미지 경량화 달성
RUN pip install --no-cache-dir -r requirements.txt

# ✨ 최적화 포인트: 이미지 빌드 단계에서 미리 모델 다운로드
# 이렇게 하면 컨테이너를 시작할 때마다 모델을 다시 받을 필요 없이 즉시 시작됨
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 프로젝트 메인 코드 복사
COPY main.py .

# FastAPI 포트 노출
EXPOSE 8000

# 컨테이너 실행 시 uvicorn으로 서버 구동
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
