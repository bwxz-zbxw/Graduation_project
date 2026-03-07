import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.sunrgbd_dataset import SUNRGBDDataset
from src.models.model import SceneUnderstandingModel

# --- Configuration ---
BATCH_SIZE = 32 # Increased from 8 to saturate GPU
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 # Reverted to 10 as per user request
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use current directory as root, don't hardcode full path
ROOT_DIR = os.getcwd() 

# --- Transforms ---
# Training transforms with augmentation (Color Jitter only as geometric requires PC sync)
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation/Test transforms (No augmentation)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train_collate_fn(batch):
    images = []
    pcs = []
    labels = []
    paths = []
    
    for item in batch:
        # Image is FloatTensor (C, H, W) range 0-255
        img = item['image'].byte() # Convert to byte for ToPILImage
        img = train_transform(img) # Apply training transforms
        
        images.append(img)
        pcs.append(item['point_cloud'])
        labels.append(item['labels'])
        paths.append(item.get('image_path', ''))
        
    images = torch.stack(images)
    pcs = torch.stack(pcs)
    
    return {'image': images, 'point_cloud': pcs, 'labels': labels, 'paths': paths}

def val_collate_fn(batch):
    images = []
    pcs = []
    labels = []
    paths = []
    
    for item in batch:
        img = item['image'].byte() # Convert to byte for ToPILImage
        img = val_transform(img) # Apply validation transforms
        
        images.append(img)
        pcs.append(item['point_cloud'])
        labels.append(item['labels'])
        paths.append(item.get('image_path', ''))
        
    images = torch.stack(images)
    pcs = torch.stack(pcs)
    
    return {'image': images, 'point_cloud': pcs, 'labels': labels, 'paths': paths}

def validate(model, loader, criterion, class_to_idx, device):
    model.eval()
    total_loss = 0
    all_preds_prob = []
    all_targets = []
    
    # Use tqdm for progress tracking during validation
    loop = tqdm(loader, total=len(loader), desc="Validating", leave=False)
    
    with torch.no_grad():
        for data in loop:
            img = data['image'].to(device)
            pc = data['point_cloud'].to(device)
            targets = encode_batch_labels(data['labels'], class_to_idx).to(device)
            
            outputs = model(img, pc)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Use Sigmoid for multi-label probabilities
            preds = torch.sigmoid(outputs)
            
            all_preds_prob.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            loop.set_postfix(val_loss=loss.item())
            
    # Concatenate all batches
    all_preds_prob = np.vstack(all_preds_prob)
    all_targets = np.vstack(all_targets)
    
    # Lower threshold to improve recall for rare classes
    threshold = 0.3 
    all_preds = (all_preds_prob > threshold).astype(int)
    
    # Calculate Metrics
    micro_f1 = f1_score(all_targets, all_preds, average='micro')
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='micro', zero_division=0)
    
    avg_loss = total_loss / len(loader)
    
    return {
        'loss': avg_loss,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall
    }

def encode_batch_labels(label_list_batch, class_to_idx):
    # label_list_batch: list of (list of strings)
    batch_target = []
    num_classes = len(class_to_idx)
    for labels in label_list_batch:
        target = torch.zeros(num_classes)
        for l in labels:
            if l in class_to_idx:
                target[class_to_idx[l]] = 1.0
        batch_target.append(target)
    return torch.stack(batch_target)

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Dataset & DataLoader
    print("Initializing Dataset...")
    full_dataset = SUNRGBDDataset(root_dir=ROOT_DIR)
    
    # Define classes (Top 37 based on paper or frequency)
    ALL_CLASSES = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
        'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 
        'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 
        'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 
        'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'
    ]
    class_to_idx = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
    num_classes = len(ALL_CLASSES)

    # Split train/val
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Determine number of workers
    num_workers = 8 if os.name == 'posix' else 0 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=train_collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=val_collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print(f"Initializing Model with {num_classes} classes...")
    model = SceneUnderstandingModel(num_classes=num_classes).to(DEVICE)
    
    # 3. Loss & Optimizer
    # Calculate Class Weights for Imbalance
    weight_path = os.path.join(ROOT_DIR, 'class_weights.pt')
    if os.path.exists(weight_path):
        print(f"Loading class weights from {weight_path}")
        pos_weight = torch.load(weight_path).to(DEVICE)
    else:
        print("Computing class weights from metadata...")
        # Iterating all metadata to count class occurrences
        image_counts = {cls: 0 for cls in ALL_CLASSES}
        for i in tqdm(range(len(full_dataset)), desc="Counting Classes"):
            # Access metadata directly (Fast)
            sample = full_dataset.all_meta[full_dataset.indices[i]]
            existing_labels = set()
            if hasattr(sample, 'groundtruth3DBB'):
                gts = sample.groundtruth3DBB
                if isinstance(gts, np.ndarray) and gts.size > 0:
                    if gts.ndim == 0: gts = np.array([gts])
                    for gt in gts:
                        try:
                            if hasattr(gt, 'classname'):
                                cls_name = gt.classname
                                # Handle numpy string arrays if any
                                if not isinstance(cls_name, str):
                                     cls_name = str(cls_name)
                                if cls_name in ALL_CLASSES:
                                    existing_labels.add(cls_name)
                        except: pass
            
            for cls in existing_labels:
                image_counts[cls] += 1
        
        pos_weights = []
        for cls in ALL_CLASSES:
            pos = image_counts[cls]
            if pos == 0: pos = 1 # Avoid div by zero
            neg = total_len - pos
            # Use Square Root to dampen extreme weights (e.g. 1000 -> 31)
            # This prevents the model from hallucinating rare classes everywhere
            weight = (neg / pos) ** 0.5 
            pos_weights.append(weight)
            
        pos_weight = torch.tensor(pos_weights).float().to(DEVICE)
        torch.save(pos_weight, weight_path)
        print("Weights saved.")

    # Use pos_weight in BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # verbose argument removed for compatibility with newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 4. Training Loop
    print("Starting Training...")
    best_macro_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        train_loss = 0
        for batch_idx, data in enumerate(loop):
            img = data['image'].to(DEVICE)
            pc = data['point_cloud'].to(DEVICE)
            targets = encode_batch_labels(data['labels'], class_to_idx).to(DEVICE)
            
            outputs = model(img, pc)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation
        val_metrics = validate(model, val_loader, criterion, class_to_idx, DEVICE)
        val_loss = val_metrics['loss']
        val_macro_f1 = val_metrics['macro_f1']
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Macro F1: {val_macro_f1:.4f}")
        print(f"  Micro F1: {val_metrics['micro_f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        
        # Step Scheduler based on Macro F1 (maximizing it)
        scheduler.step(val_macro_f1)
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            save_path = os.path.join(ROOT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (Macro F1) to {save_path}")

if __name__ == '__main__':
    train()
