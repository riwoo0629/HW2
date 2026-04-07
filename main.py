from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import io

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="A simple API for object detection using YOLOv8n. Includes JSON counting and Image plotting.",
    version="1.1.0"
)

# Load the lightweight YOLOv8n model
# It will be downloaded automatically the first time if not present in the directory
model = YOLO("yolov8n.pt")

@app.get("/")
def read_root():
    return {"message": "Welcome to YOLOv8 Object Detection API. Please use /detect/ or /detect/image/ to analyze an image."}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    사진을 업로드하면 탐지된 각 객체의 갯수 합계와 세부 정보를 JSON 텍스트로 알려줍니다.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 추론 수행
        results = model(image)

        detected_objects = []
        object_counts = Counter()

        # 분석 결과 파싱
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                confidence = float(box.conf[0].item())
                
                detected_objects.append({
                    "object": class_name,
                    "confidence": round(confidence, 4)
                })
                # 동일한 객체의 출현 횟수 계산용
                object_counts[class_name] += 1

        return {
            "filename": file.filename, 
            "total_detected": len(detected_objects),
            "object_counts": dict(object_counts),  # {"person": 3, "car": 1} 형식으로 출력
            "detections": detected_objects
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/image/")
async def detect_objects_and_return_image(file: UploadFile = File(...)):
    """
    사진을 업로드하면 바운딩 박스(네모칸)가 실제로 그려진 그림 데이터를 
    직접 (StreamingResponse 형식으로) 곧장 반환해 줍니다.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 추론 수행
        results = model(image)
        
        # 첫 번째 분석 결과에서 그림이 그려진 매트릭스 추출
        annotated_img_array = results[0].plot()

        # YOLO 모델은 기본적으로 BGR(파랑-초록-빨강) 색상 배열로 그림을 반환함
        # 이것을 PIL 이미지로 정상적인 RGB 색상으로 뒤집어서 변환
        annotated_image = Image.fromarray(annotated_img_array[..., ::-1])

        # 이미지를 담을 빈 바이트 버퍼 메모리 통 생성
        img_buffer = io.BytesIO()
        
        # 메모리 버퍼에 JPEG 형태로 이미지 꾸역꾸역 집어넣기
        annotated_image.save(img_buffer, format="JPEG")
        
        # 파일 커서를 0으로 되돌려야 처음부터 데이터를 보낼 수 있음
        img_buffer.seek(0)
        
        # 클라이언트 브라우저로 이미지 데이터를 직접 쏘아서 눈에 보이게 해줌
        return StreamingResponse(img_buffer, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
