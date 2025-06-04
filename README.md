# Badminton Shot Classification

Classify badminton shots from video using pretrained video models like **Swin3D-T** and **R3D-18** (Kinetics-400).

The dataset used was shuttleset which was preprocessed to create shot clips from rallies.

### Train & evaluation

```
python train_swin3d.py     
train_r3d.py
```

### Infer

```
python infer.py --model swin3d_t --ckpt swin3d_badminton_final.pth --video path/to/video.mp4
```

### Weights

https://drive.google.com/drive/folders/19fpinc5C8T9sVYUDgUS20PXhtztT40Vc?usp=drive_link

## Teammates

Ayan Kashyap
Anuska Maity
Pratyusha Mitra
