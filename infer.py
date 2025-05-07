import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.video import swin3d_t, Swin3D_T_Weights, r3d_18
import cv2
import numpy as np
from action_dataset import BadmintonTemporalShotDataset  # for idx_to_shot_type
import os

def load_model(model_name, num_classes, ckpt_path, device):
    if model_name == "swin3d_t":
        model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == "r3d_18":
        model = r3d_18(pretrained=True)
        model.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()

def read_video_clip(video_path, clip_len=16, resize=(128,128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // clip_len, 1)

    for i in range(clip_len):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # If fewer frames, pad with last
    while len(frames) < clip_len:
        frames.append(frames[-1])
    
    frames = np.stack(frames)  # [T, H, W, C]
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [T,C,H,W]
    norm = T.Normalize(mean=[0.45]*3, std=[0.225]*3)
    frames = torch.stack([norm(f) for f in frames])  # [T,C,H,W]
    return frames.permute(1,0,2,3).unsqueeze(0)  # [1,C,T,H,W]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["swin3d_t", "r3d_18"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19 # Adjust based on your dataset
    shot_type_to_idx = {
                'net shot': 1,
                'return net': 2,
                'smash': 3,
                'wrist smash': 4,
                'lob': 5,
                'defensive return lob': 6,
                'clear': 7,
                'drive': 8,
                'driven flight': 9,
                'back-court drive': 10,
                'drop': 11,
                'passive drop': 12,
                'push': 13,
                'rush': 14,
                'defensive return drive': 15,
                'cross-court net shot': 16,
                'short service': 17,
                'long service': 18,
                'unknown': 19
            }
    idx_to_label = {v: k for k, v in shot_type_to_idx.items()}
    model = load_model(args.model, num_classes, args.ckpt, device)
    clip = read_video_clip(args.video).to(device)
    with torch.no_grad():
        logits = model(clip)
        pred = logits.argmax(dim=1).item()

    print(f"Predicted class: {idx_to_label[pred]}")

if __name__ == "__main__":
    main()
