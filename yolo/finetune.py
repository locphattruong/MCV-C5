from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo12x.pt")

# Train on the new dataset
model.train(
    data='kitti_mots_finetune.yaml',
    pretrained = True,
    epochs=30,
    imgsz=640,
    batch=8,
    name='kitti_mots_finetune',
    plots = True,
    device = [0]
)