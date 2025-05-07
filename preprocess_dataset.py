import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from collections import defaultdict
from moviepy import VideoFileClip
import tempfile
from IPython.display import Video as IPythonVideo
import tqdm

DATA_DIR = "/home/ayan/work/Badminton/Shuttleset_dataset"
DATASET_DIR = "/home/ayan/work/Badminton/ShuttleSet/set"
MATCH_CSV = os.path.join(DATASET_DIR, "match.csv")
HOMOGRAPHY_CSV = os.path.join(DATASET_DIR, "match.csv")
shot_types_mapping = {
    '放小球': 'net shot',
    '擋小球': 'return net',
    '殺球': 'smash',
    '點扣': 'wrist smash',
    '挑球': 'lob',
    '防守回挑': 'defensive return lob',
    '長球': 'clear',
    '平球': 'drive',
    '小平球': 'driven flight',
    '後場抽平球': 'back-court drive',
    '切球': 'drop',
    '過渡切球': 'passive drop',
    '推球': 'push',
    '撲球': 'rush',
    '防守回抽': 'defensive return drive',
    '勾球': 'cross-court net shot',
    '發短球': 'short service',
    '發長球': 'long service',
    '未知球種': 'unknown'  # Added for completeness
}
ALL_CSVS = defaultdict(dict)
ALL_VIDEOS = {}

def process_badminton_videos(ALL_VIDEOS, ALL_CSVS, output_dir="processed_data", sample_rate=5):
    """
    Process badminton videos and organize frames by rally with subsampling
    
    Args:
        ALL_VIDEOS: Dictionary mapping video names to file paths
        ALL_CSVS: Dictionary mapping video names to lists of CSV file paths
        output_dir: Directory to save processed data
        sample_rate: Sample rate for frame extraction (e.g., 5 means save every 5th frame)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for name, video_path in tqdm(ALL_VIDEOS.items(), desc="Processing videos"):
        print(f"Video name: {name}, Video path: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS: {fps}")
        sets_per_video = ALL_CSVS[name]
        
        # Process each set separately to maintain correct rally_id
        for set_idx in range(len(sets_per_video)):
        # for set_idx, set_csv_path in enumerate(sets_per_video):
            set_csv_path = sets_per_video[set_idx]
            print(f"Processing set {set_idx + 1} from {set_csv_path}")
            
            # Read the set data
            set_df = pd.read_csv(set_csv_path)
            
            # Clean any N/A or NaN shots
            set_df = set_df.dropna(subset=['type'])
            
            # Convert frame_num to int to ensure correct processing
            set_df['frame_num'] = set_df['frame_num'].astype(int)
            
            # Map shot types
            set_df['type'] = set_df['type'].map(shot_types_mapping)
            
            # Drop rows where mapping failed (unknown shot types)
            set_df = set_df[set_df['type'].notna()]
            
            # Group by rally for this set
            rallies = set_df.groupby('rally')
            
            print(f"Found {len(rallies)} rallies in set {set_idx + 1}")
            
            # Process each rally in this set
            for rally_id, rally_data in tqdm(rallies, desc=f"Processing rallies for {name} set {set_idx + 1}"):
                # Create a unique rally identifier that includes the set
                unique_rally_id = f"{name}_set{set_idx+1}_rally{rally_id}"
                
                # Create rally directory
                rally_dir = os.path.join(output_dir, unique_rally_id)
                frames_dir = os.path.join(rally_dir, "frames")
                os.makedirs(frames_dir, exist_ok=True)
                
                # Get rally frame range
                rally_start_frame = rally_data['frame_num'].min() - 30  # Small buffer before rally starts
                rally_end_frame = rally_data['frame_num'].max() + 30    # Small buffer after rally ends
                rally_start_frame = max(0, rally_start_frame)  # Ensure not negative
                
                # Create a set of all shot frames for quick lookup
                shot_frames = set(rally_data['frame_num'].tolist())
                
                # Set up for frame extraction
                cap.set(cv2.CAP_PROP_POS_FRAMES, rally_start_frame)
                current_frame = rally_start_frame
                frame_metadata = []
                
                # Extract frames with subsampling, always including shot frames
                while current_frame <= rally_end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Save this frame if it's a shot frame or if it falls on the sampling rate
                    is_shot_frame = current_frame in shot_frames
                    should_sample = (current_frame - rally_start_frame) % sample_rate == 0
                    
                    if is_shot_frame or should_sample:
                        # Save the frame
                        frame_file = f"frame_{current_frame:06d}.jpg"
                        frame_path = os.path.join(frames_dir, frame_file)
                        cv2.imwrite(frame_path, frame)
                        
                        # Add to metadata
                        if is_shot_frame:
                            # This is a shot frame - get its data
                            shot_data = rally_data[rally_data['frame_num'] == current_frame].iloc[0]
                            frame_metadata.append({
                                'frame_file': frame_file,
                                'frame_num': int(current_frame),  # Ensure frame_num is an integer
                                'is_shot_frame': True,
                                'player': shot_data['player'],
                                'shot_type': shot_data['type'],
                                'ball_round': int(shot_data['ball_round']) if 'ball_round' in shot_data else None,
                                'landing_x': float(shot_data['landing_x']) if 'landing_x' in shot_data else None,
                                'landing_y': float(shot_data['landing_y']) if 'landing_y' in shot_data else None,
                            })
                        else:
                            # This is a regular sampled frame (not a shot)
                            frame_metadata.append({
                                'frame_file': frame_file,
                                'frame_num': int(current_frame),  # Ensure frame_num is an integer
                                'is_shot_frame': False,
                                'player': None,
                                'shot_type': 'no_shot',  # Explicitly mark as "no_shot"
                                'ball_round': None,
                                'landing_x': None,
                                'landing_y': None,
                            })
                    
                    # Move to next frame
                    current_frame += 1
                
                # Save metadata to CSV
                metadata_df = pd.DataFrame(frame_metadata)
                metadata_df.to_csv(os.path.join(rally_dir, "metadata.csv"), index=False)
                
                # Also save original rally data
                rally_data.to_csv(os.path.join(rally_dir, "rally_data.csv"), index=False)
                
                # Save rally video clip
                output_clip_path = os.path.join(rally_dir, "rally_clip.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                # Get frame dimensions
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                out = cv2.VideoWriter(output_clip_path, fourcc, fps, (width, height))
                
                # Reset to start of rally
                cap.set(cv2.CAP_PROP_POS_FRAMES, rally_start_frame)
                current_frame = rally_start_frame
                
                # Write frames to video
                while current_frame <= rally_end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    current_frame += 1
                
                out.release()
        
        # Release video capture when done with this video
        cap.release()