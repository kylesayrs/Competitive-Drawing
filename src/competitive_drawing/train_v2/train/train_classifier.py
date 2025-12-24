from competitive_drawing.train_v2.dataset.dataset import QuickDrawDataset

dataset = QuickDrawDataset(
    "src/competitive_drawing/train_v2/camera.ndjson",
    "src/competitive_drawing/train_v2/coffee_cup.ndjson",
)

dataset