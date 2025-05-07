import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
from torchvision import transforms
import glob


class BadmintonTemporalShotDataset(Dataset):
    """
    Dataset for badminton temporal shot classification.
    
    Each sample is a sequence of frames representing a single badminton shot.
    """
    
    def __init__(self, 
                 rally_dirs,  # List of rally directories to use
                 clip_length=16,  # Number of frames per clip
                 transform=None,
                 temporal_transform=None,
                 frame_sampling='uniform',  # 'uniform' or 'random'
                 frame_skip=1,  # Take every nth frame
                 return_metadata=False):
        """
        Args:
            rally_dirs: List of rally directories to use
            clip_length: Number of frames to include in each clip
            transform: Spatial transformations to apply to frames
            temporal_transform: Temporal transformations to apply to frame sequences
            frame_sampling: Method for sampling frames ('uniform' or 'random')
            frame_skip: Sample every n-th frame
            return_metadata: Whether to return metadata with samples
        """
        self.rally_dirs = rally_dirs
        self.clip_length = clip_length
        self.transform = transform
        self.temporal_transform = temporal_transform
        self.frame_sampling = frame_sampling
        self.frame_skip = frame_skip
        self.return_metadata = return_metadata
        
        # Use the provided shot type mapping but exclude 'no_shot'
        self.shot_type_to_idx = {
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
        # 
        shot_types = list(self.shot_type_to_idx.keys())
        
        self.shot_type_to_idx = {t:i for i,t in enumerate(shot_types)}
        self.idx_to_shot_type = {i:t for i,t in enumerate(shot_types)}
        self.num_classes = len(shot_types)  # now labels span 0…18

        
        # Build shot sequences list
        self.shot_sequences = []
        
        for rally_dir in self.rally_dirs:
            rally_name = os.path.basename(rally_dir)
            metadata_path = os.path.join(rally_dir, "metadata.csv")
            
            if not os.path.exists(metadata_path):
                continue
                
            # Read metadata
            metadata = pd.read_csv(metadata_path)
            
            # Group frames by shot
            shot_frames = []
            current_shot = None
            current_shot_frames = []
            current_shot_data = None
            
            # Sort by frame number to ensure temporal ordering
            metadata = metadata.sort_values(by='frame_num')
            
            for _, row in metadata.iterrows():
                if row['is_shot_frame'] and row['shot_type'] != 'no_shot' and row['shot_type'] in self.shot_type_to_idx:
                    # If this is a new shot type or a new player
                    if (current_shot != row['shot_type'] or 
                        (current_shot_data is not None and current_shot_data['player'] != row['player'])):
                        
                        # Save the previous shot sequence if it exists
                        if current_shot is not None and len(current_shot_frames) > 0:
                            shot_frames.append({
                                'rally_name': rally_name,
                                'shot_type': current_shot,
                                'player': current_shot_data['player'],
                                'frames': current_shot_frames.copy(),
                                'ball_round': current_shot_data['ball_round'],
                                'landing_x': current_shot_data['landing_x'],
                                'landing_y': current_shot_data['landing_y']
                            })
                        
                        # Start a new shot sequence
                        current_shot = row['shot_type']
                        current_shot_frames = []
                        current_shot_data = row
                    
                    # Add this frame to the current shot sequence
                    frame_path = os.path.join(rally_dir, "frames", row['frame_file'])
                    if os.path.exists(frame_path):
                        current_shot_frames.append({
                            'path': frame_path,
                            'frame_num': row['frame_num']
                        })
                
                # For non-shot frames, add them to the current shot sequence if we're in one
                elif current_shot is not None:
                    frame_path = os.path.join(rally_dir, "frames", row['frame_file'])
                    if os.path.exists(frame_path):
                        # Find the frames that are temporally between this shot and the next shot
                        # Only include frames that come after the current shot's first frame
                        if row['frame_num'] > current_shot_frames[0]['frame_num']:
                            current_shot_frames.append({
                                'path': frame_path,
                                'frame_num': row['frame_num']
                            })
            
            # Don't forget the last shot sequence in the rally
            if current_shot is not None and len(current_shot_frames) > 0:
                shot_frames.append({
                    'rally_name': rally_name,
                    'shot_type': current_shot,
                    'player': current_shot_data['player'],
                    'frames': current_shot_frames.copy(),
                    'ball_round': current_shot_data['ball_round'],
                    'landing_x': current_shot_data['landing_x'],
                    'landing_y': current_shot_data['landing_y']
                })
            
            # Add valid shot sequences (with enough frames) to the dataset
            for shot in shot_frames:
                if len(shot['frames']) >= min(3, self.clip_length):  # Require at least 3 frames or clip_length
                    self.shot_sequences.append(shot)
            
        print(f"BadmintonTemporalShotDataset created with {len(self.shot_sequences)} sequences")
        # Count samples per class
        shot_counts = {}
        for seq in self.shot_sequences:
            shot_type = seq['shot_type']
            if shot_type not in shot_counts:
                shot_counts[shot_type] = 0
            shot_counts[shot_type] += 1
        print(f"Samples per class: {shot_counts}")
    
    def __len__(self):
        return len(self.shot_sequences)
    
    def __getitem__(self, idx):
        """
        Returns a clip of frames representing a single shot with its label.
        
        Returns:
            frames: Tensor of shape [clip_length, channels, height, width]
            label: Integer label of the shot type
            metadata (optional): Dictionary with shot metadata
        """
        shot_data = self.shot_sequences[idx]
        
        frames_data = shot_data['frames']
        frames_data = sorted(frames_data, key=lambda x: x['frame_num'])
        
        # Apply frame skipping
        frames_data = frames_data[::self.frame_skip]
        
        # Sample frames according to the specified method
        if len(frames_data) >= self.clip_length:
            if self.frame_sampling == 'uniform':
                # Take uniformly spaced frames
                indices = np.linspace(0, len(frames_data) - 1, self.clip_length, dtype=int)
                sampled_frames_data = [frames_data[i] for i in indices]
            elif self.frame_sampling == 'random':
                # Randomly sample frames
                if self.temporal_transform:
                    indices = self.temporal_transform(list(range(len(frames_data))), self.clip_length)
                    sampled_frames_data = [frames_data[i] for i in indices]
                else:
                    indices = np.random.choice(len(frames_data), self.clip_length, replace=False)
                    indices.sort()
                    sampled_frames_data = [frames_data[i] for i in indices]
            else:
                # Default to taking the first clip_length frames
                sampled_frames_data = frames_data[:self.clip_length]
        else:
            # If we don't have enough frames, duplicate the last frame
            sampled_frames_data = frames_data.copy()
            while len(sampled_frames_data) < self.clip_length:
                sampled_frames_data.append(sampled_frames_data[-1])
        
        # Load frames
        frames = []
        for frame_data in sampled_frames_data:
            img = Image.open(frame_data['path']).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        # Stack frames
        if isinstance(frames[0], torch.Tensor):
            frames = torch.stack(frames)
        else:
            frames = np.stack(frames)
        
        # Get label
        label = self.shot_type_to_idx[shot_data['shot_type']]
        
        if self.return_metadata:
            metadata = {
                'rally_name': shot_data['rally_name'],
                'shot_type': shot_data['shot_type'],
                'player': shot_data['player'],
                'ball_round': shot_data['ball_round'],
                'landing_x': shot_data['landing_x'],
                'landing_y': shot_data['landing_y'],
                'frames': [f['frame_num'] for f in sampled_frames_data]
            }
            return frames, label, metadata
        
        return frames, label


# Example usage:
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    import os
    
    # Define transformations
    spatial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get rally directories
    data_dir = 'processed_data'
    all_rally_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))]
    
    # Filter to only include rallies with metadata
    valid_rally_dirs = [d for d in all_rally_dirs if os.path.exists(os.path.join(d, "metadata.csv"))]
    
    # Split the rally directories
    train_dirs, test_dirs = train_test_split(
        valid_rally_dirs, 
        test_size=0.2, 
        random_state=42
    )
    
    # Create datasets
    train_dataset = BadmintonTemporalShotDataset(
        rally_dirs=train_dirs,
        clip_length=16,
        transform=spatial_transform,
        frame_sampling='uniform'
    )
    
    test_dataset = BadmintonTemporalShotDataset(
        rally_dirs=test_dirs,
        clip_length=16,
        transform=spatial_transform,
        frame_sampling='uniform'
    )
    
    # Example: get a sample
    frames, label = train_dataset[0]
    print(f"Sample shapes: {frames.shape}, Label: {label} ({train_dataset.idx_to_shot_type[label]})")