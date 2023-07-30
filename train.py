import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchmetrics
from data import SegmentationDataset, get_transforms
from model import RRDB_Fuzzy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 75
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_MEAN = [0.2826, 0.2826, 0.2826]
IMAGE_STD = [0.2695, 0.2695, 0.2695]
LEARNING_RATE = 1e-4
TRAIN_IMG_DIR = "/kaggle/input/ra-attention-util-notebookf6efc8202c/frames/"
TRAIN_MASK_DIR = "/kaggle/input/ra-attention-util-notebookf6efc8202c/masks/"
print("Number of samples:", len(os.listdir(TRAIN_IMG_DIR)))
train_img_files, test_img_files = train_test_split(os.listdir(TRAIN_IMG_DIR), test_size=0.2, shuffle=True, random_state=42)

train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_img_files, get_transforms(True, IMAGE_HEIGHT, IMAGE_WIDTH))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
test_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, test_img_files, get_transforms(False, IMAGE_HEIGHT, IMAGE_WIDTH))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

model = RRDB_Fuzzy().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

train_metrics = [torchmetrics.Accuracy().to(DEVICE), torchmetrics.Dice().to(DEVICE), torchmetrics.classification.BinaryJaccardIndex().to(DEVICE)]
test_metrics = [torchmetrics.Accuracy().to(DEVICE), torchmetrics.Dice().to(DEVICE), torchmetrics.classification.BinaryJaccardIndex().to(DEVICE)]

max_dice = 0
max_epoch = 0
NUM_EPOCHS = 3
for epoch in range(1, NUM_EPOCHS + 1):
    print(epoch, "/", NUM_EPOCHS)
    total_loss = 0
    model.train()
    for imgs, masks in tqdm.notebook.tqdm(train_loader):
        optimizer.zero_grad()
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = torch.sigmoid(model(imgs))
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        for metric in train_metrics:
            metric.update(outputs, masks.to(torch.int8))

    print("Train Loss:", total_loss)
    for metric in train_metrics:
        print(metric, metric.compute().item())
        metric.reset()

    print()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm.notebook.tqdm(test_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = torch.sigmoid(model(imgs))
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            for metric in test_metrics:
                metric.update(outputs, masks.to(torch.int8))

    print("Test Loss:", total_loss)
    for metric in test_metrics:
        print(metric, metric.compute().item())
        if metric.__str__() == "Dice()" and metric.compute().item() > max_dice:
            max_dice = metric.compute().item()
            max_epoch = epoch
            torch.save(model, f"model-{epoch}.pt")
        metric.reset()
    print()

    scheduler.step()

print("Max Dice:", max_dice, "Epoch:", max_epoch)
