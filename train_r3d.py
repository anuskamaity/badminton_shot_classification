import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from torchvision import transforms
from sklearn.model_selection import train_test_split
from action_dataset import BadmintonTemporalShotDataset  # your dataset
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Initialize wandb
wandb.init(project='badminton-shot-classification', config={
    'clip_length': 16,
'batch_size': 32,
    'epochs': 10,
    'lr': 1e-4,
    'frame_skip': 1,
    'frame_sampling': 'uniform',
})

config = wandb.config

# Standard image transforms
spatial_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
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

# Datasets
train_dataset = BadmintonTemporalShotDataset(
    rally_dirs=train_dirs,  # Replace
    clip_length=config.clip_length,
    transform=spatial_transform,
    frame_skip=config.frame_skip,
    frame_sampling=config.frame_sampling,
    return_metadata=False
)

val_dataset = BadmintonTemporalShotDataset(
    rally_dirs=val_dirs,  # Replace
    clip_length=config.clip_length,
    transform=spatial_transform,
    frame_skip=config.frame_skip,
    frame_sampling=config.frame_sampling,
    return_metadata=False
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

# Model: r3d_18 pretrained on Kinetics
model = r3d_18(pretrained=True)
model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))  # Support clip len != 16
model.fc = nn.Linear(model.fc.in_features, train_dataset.num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

cfg = wandb.config
# 5. Training & Validation with seaborn plots
for epoch in range(cfg.epochs):
    # --- Training ---
    model.train()
    running_correct, running_total = 0, 0
    batch_losses = []
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
    for clips, labels in train_pbar:
        clips = clips.permute(0,2,1,3,4).to(device)
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

    # Plot per-batch training loss for this epoch
    plt.figure(figsize=(8,4))
    sns.lineplot(x=list(range(len(batch_losses))), y=batch_losses)
    plt.title(f'Epoch {epoch+1} Training Loss (per batch)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'train_loss_epoch{epoch+1}.png')
    wandb.log({f"train_loss_batch_epoch{epoch+1}": wandb.Image(plt)}, step=epoch+1)
    plt.close()

    # --- Validation ---
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

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(train_dataset.num_classes)))

    # Plot seaborn heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=[train_dataset.idx_to_shot_type[i] for i in range(train_dataset.num_classes)],
                yticklabels=[train_dataset.idx_to_shot_type[i] for i in range(train_dataset.num_classes)])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f'Epoch {epoch+1} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_epoch{epoch+1}.png')
    wandb.log({f"confusion_matrix_epoch{epoch+1}": wandb.Image(plt)}, step=epoch+1)
    plt.close()

    # Log scalar aggregates
    wandb.log({
        'train/acc': running_correct/running_total,
        'train/loss': np.mean(batch_losses),
        'val/acc': acc,
        'val/loss': np.mean(val_losses),
        'epoch': epoch+1
    })

# 6. Save final model
torch.save(model.state_dict(), "r3d_badminton_final.pth")
wandb.save("r3d_badminton_final.pth")