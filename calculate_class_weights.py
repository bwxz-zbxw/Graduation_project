import os
import numpy as np
import torch
import torch.nn as nn
from src.data.sunrgbd_dataset import SUNRGBDDataset
from tqdm import tqdm
from collections import Counter

# Configuration
ROOT_DIR = os.getcwd()
BATCH_SIZE = 32

def calculate_weights():
    print("Initializing Dataset...")
    # Use split='train' ideally, but dataset class uses all for now
    dataset = SUNRGBDDataset(root_dir=ROOT_DIR)
    
    print("Iterating over dataset to count labels...")
    all_labels = []
    
    # We iterate manually to avoid loading images
    for idx in tqdm(range(len(dataset))):
        sample = dataset.all_meta[dataset.indices[idx]]
        
        sample_labels = []
        if hasattr(sample, 'groundtruth3DBB'):
            gts = sample.groundtruth3DBB
            
            # Handle different types (sometimes it's valid, sometimes empty, sometimes array)
            if isinstance(gts, np.ndarray) and gts.size > 0:
                if gts.ndim == 0:
                    gts = np.array([gts])
                    
                for gt in gts:
                    try:
                        # classname might be a string or array
                        if hasattr(gt, 'classname'):
                            classname = gt.classname
                            if isinstance(classname, str):
                                sample_labels.append(classname)
                            elif isinstance(classname, np.ndarray):
                                # Sometimes it's an array of encoded chars
                                sample_labels.append(str(classname))
                    except:
                        pass
        
        all_labels.extend(sample_labels)

    print(f"Total labeled objects found: {len(all_labels)}")
    
    # Define classes (Must match train.py)
    ALL_CLASSES = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
        'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 
        'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 
        'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 
        'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'
    ]
    
    # Count frequencies
    counts = Counter(all_labels)
    
    # Filter for relevant classes
    class_counts = {cls: counts.get(cls, 0) for cls in ALL_CLASSES}
    
    print("\nClass Counts:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")
        
    # Calculate pos_weight for BCEWithLogitsLoss
    # pos_weight = num_negatives / num_positives
    # num_positives = class_count
    # num_negatives = total_samples - class_count (Wait, usually total_samples, but here it's multi-label per image)
    # Actually, for BCE loss per class:
    # N is total number of images (samples)
    # pos = number of images containing class C
    # neg = N - pos
    # weight = neg / pos
    
    # We need to count frequency per IMAGE, not total objects
    # Let's redo the loop to count per image
    
    image_counts = {cls: 0 for cls in ALL_CLASSES}
    total_images = len(dataset)
    
    print("\nRecalculating per-image frequency...")
    for idx in tqdm(range(len(dataset))):
        sample = dataset.all_meta[dataset.indices[idx]]
        existing_labels = set()
        
        if hasattr(sample, 'groundtruth3DBB'):
            gts = sample.groundtruth3DBB
            if isinstance(gts, np.ndarray) and gts.size > 0:
                if gts.ndim == 0:
                    gts = np.array([gts])  
                for gt in gts:
                    try:
                        if hasattr(gt, 'classname'):
                            cls = gt.classname
                            if cls in ALL_CLASSES:
                                existing_labels.add(cls)
                    except:
                        pass
        
        for cls in existing_labels:
            image_counts[cls] += 1
            
    print("\nImage Frequency per Class:")
    pos_weights = []
    sorted_classes = sorted(ALL_CLASSES) # train.py uses ALL_CLASSES list order, not sorted!
    # train.py: class_to_idx = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
    # So we must iterate ALL_CLASSES in order
    
    for cls in ALL_CLASSES:
        pos = image_counts[cls]
        if pos == 0:
            print(f"Warning: Class {cls} has 0 samples! Setting weight to 1.")
            weight = 1.0
        else:
            neg = total_images - pos
            weight = neg / pos
        
        pos_weights.append(weight)
        print(f"{cls}: {pos} images (Weight: {weight:.4f})")
        
    pos_weights_tensor = torch.tensor(pos_weights).float()
    
    print(f"\nSaving weights to class_weights.pt...")
    torch.save(pos_weights_tensor, 'class_weights.pt')
    print("Done.")

if __name__ == '__main__':
    calculate_weights()
