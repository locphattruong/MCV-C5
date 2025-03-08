from ultralytics import YOLO
import wandb

wandb.init(project="YOLO")
# Initialize the YOLO model
model = YOLO("yolo12x.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="kitti_mots_finetune.yaml",
    epochs=30,
    iterations=300,
    optimizer="AdamW",
    plots=True,
    save=True,
    val=False,
)