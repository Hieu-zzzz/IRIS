from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data='TRAIN.v2i.yolov8/data.yaml',
    epochs=100,
    imgsz=312,
    batch=16,
    name='drink_detection',
    patience=50,
    save=True,
    plots=True
)

# Validate the model
metrics = model.val()

print(f"Training completed!")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
