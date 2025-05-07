import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import swin3d_t, Swin3D_T_Weights  # Video SwinTiny
from torchvision import transforms
from action_dataset import BadmintonTemporalShotDataset
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. W&B Setup ---
wandb.init(project='badminton-shot-swin3d', config={
    'model': 'swin3d_t',
    'weights': 'KINETICS400_V1',
    'clip_length': 16,
    'batch_size': 16,
    'epochs': 10,
    'lr': 1e-4,
    'frame_skip': 1,
    'frame_sampling': 'uniform',
})
cfg = wandb.config

# --- 2. Transforms (same as before) ---
spatial_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225])
])
data_dir = 'test_rally'
all_rally_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))]

# Filter to only include rallies with metadata
valid_rally_dirs = [d for d in all_rally_dirs if os.path.exists(os.path.join(d, "metadata.csv"))]

# Split the rally directories
train_dirs, val_dirs = train_test_split(
    valid_rally_dirs, 
    test_size=0.2, 
    random_state=42
)
# --- 3. Data ---
train_ds = BadmintonTemporalShotDataset(
    rally_dirs=train_dirs,
    clip_length=cfg.clip_length,
    transform=spatial_transform,
    frame_skip=cfg.frame_skip,
    frame_sampling=cfg.frame_sampling,
)
val_ds = BadmintonTemporalShotDataset(
    rally_dirs=val_dirs,
    clip_length=cfg.clip_length,
    transform=spatial_transform,
    frame_skip=cfg.frame_skip,
    frame_sampling=cfg.frame_sampling,
)
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=4)

# --- 4. Model, Loss, Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load Video Swin Tiny with Kinetics-400 weights :contentReference[oaicite:2]{index=2}
model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)  
# Replace head for your num_classes
print(model)
model.head = nn.Linear(model.head.in_features, train_ds.num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
wandb.watch(model, log="all", log_freq=10)

# --- 5. Train & Validate with tqdm + seaborn plots ---
for epoch in range(cfg.epochs):
    # ---- Training ----
    model.train()
    running_correct, running_total = 0, 0
    batch_losses = []
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
    for clips, labels in train_pbar:
        clips = clips.permute(0,2,1,3,4).to(device)  # [B, T, C, H, W] → [B, C, T, H, W]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)
        train_acc = running_correct / running_total
        train_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_acc:.4f}")

    plt.figure(figsize=(8,4))
    sns.lineplot(x=list(range(len(batch_losses))), y=batch_losses)
    plt.title(f'Epoch {epoch+1} Training Loss (per batch) - Swin3D_T')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'train_loss_epoch{epoch+1}_swin.png')
    wandb.log({f"train_loss_batch_epoch{epoch+1}": wandb.Image(plt)}, step=epoch+1)
    plt.close()

    model.eval()
    val_losses, all_preds, all_labels = [], [], []
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]  ")
    with torch.no_grad():
        for clips, labels in val_pbar:
            clips = clips.permute(0,2,1,3,4).to(device)
            labels = labels.to(device)

            outputs = model(clips)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            acc = (np.array(all_preds) == np.array(all_labels)).mean()
            val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    # Confusion matrix & heatmap via seaborn :contentReference[oaicite:4]{index=4}
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(train_ds.num_classes)))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=[train_ds.idx_to_shot_type[i] for i in range(train_ds.num_classes)],
                yticklabels=[train_ds.idx_to_shot_type[i] for i in range(train_ds.num_classes)])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f'Epoch {epoch+1} Confusion Matrix - Swin3D_T')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_epoch{epoch+1}_swin.png')
    wandb.log({f"confusion_matrix_epoch{epoch+1}": wandb.Image(plt)}, step=epoch+1)
    plt.close()

    # Log scalars
    wandb.log({
        'train/acc': running_correct/running_total,
        'train/loss': np.mean(batch_losses),
        'val/acc': acc,
        'val/loss': np.mean(val_losses),
        'epoch': epoch+1
    })

torch.save(model.state_dict(), "swin3d_badminton_final.pth")
wandb.save("swin3d_badminton_final.pth")
