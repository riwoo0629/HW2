from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="A simple API for object detection using YOLOv8n",
    version="1.0.0"
)

# Load the lightweight YOLOv8n model
# It will be downloaded automatically the first time if not present in the directory
model = YOLO("yolov8n.pt")

@app.get("/")
def read_root():
    return {"message": "Welcome to YOLOv8 Object Detection API. Send a POST request to /detect/ with an image file."}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the image file into memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Perform inference
        results = model(image)

        # Parse results
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract class id and name
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                confidence = float(box.conf[0].item())
                
                detected_objects.append({
                    "object": class_name,
                    "confidence": round(confidence, 4)
                })

        return {
            "filename": file.filename, 
            "detections": detected_objects,
            "total_detected": len(detected_objects)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
