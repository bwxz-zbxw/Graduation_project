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
from train import train_collate_fn, val_collate_fn, encode_batch_labels

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = os.getcwd() 

def validate(model, loader, criterion, class_to_idx, device):
    model.eval()
    total_loss = 0
    all_preds_prob = []
    all_targets = []
    
    loop = tqdm(loader, total=len(loader), desc="Validating Baseline", leave=False)
    
    with torch.no_grad():
        for data in loop:
            img = data['image'].to(device)
            pc = data['point_cloud'].to(device)
            targets = encode_batch_labels(data['labels'], class_to_idx).to(device)
            
            outputs = model(img, pc)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            
            all_preds_prob.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            loop.set_postfix(val_loss=loss.item())
            
    all_preds_prob = np.vstack(all_preds_prob)
    all_targets = np.vstack(all_targets)
    
    threshold = 0.5 
    all_preds = (all_preds_prob > threshold).astype(int)
    
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

def train_baseline():
    print(f"--- Starting Ablation Study: Baseline Model (No Transformer) ---")
    print(f"Using device: {DEVICE}")
    
    # 1. Dataset & DataLoader
    full_dataset = SUNRGBDDataset(root_dir=ROOT_DIR)
    
    ALL_CLASSES = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
        'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 
        'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 
        'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 
        'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'
    ]
    class_to_idx = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
    num_classes = len(ALL_CLASSES)

    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
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
    
    # 2. Model (Baseline: use_transformer=False)
    print(f"Initializing Baseline Model (use_transformer=False)...")
    model = SceneUnderstandingModel(num_classes=num_classes, use_transformer=False).to(DEVICE)
    
    # 3. Weights
    weight_path = os.path.join(ROOT_DIR, 'class_weights.pt')
    if os.path.exists(weight_path):
        print(f"Loading class weights from {weight_path}")
        pos_weight = torch.load(weight_path).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print("Warning: Class weights not found! Computing them now...")
        # Fallback to computing weights if file is missing (e.g. fresh environment)
        # This duplicates logic from train.py but ensures robustness
        image_counts = {cls: 0 for cls in ALL_CLASSES}
        total_len = len(full_dataset)
        
        # Quick approximation or better yet, error out to force user to run train.py first?
        # Better: run calculation loop if missing
        for i in tqdm(range(len(full_dataset)), desc="Counting Classes"):
            sample = full_dataset.all_meta[full_dataset.indices[i]]
            existing_labels = set()
            if hasattr(sample, 'groundtruth3DBB'):
                gts = sample.groundtruth3DBB
                if isinstance(gts, np.ndarray) and gts.size > 0:
                    if gts.ndim == 0: gts = np.array([gts])
                    for gt in gts:
                        try:
                            if hasattr(gt, 'classname'):
                                cls_name = str(gt.classname)
                                if cls_name in ALL_CLASSES:
                                    existing_labels.add(cls_name)
                        except: pass
            for cls in existing_labels:
                image_counts[cls] += 1
        
        pos_weights = []
        for cls in ALL_CLASSES:
            pos = image_counts[cls]
            if pos == 0: pos = 1 
            neg = total_len - pos
            weight = (neg / pos) ** 0.5 
            pos_weights.append(weight)
            
        pos_weight = torch.tensor(pos_weights).float().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        torch.save(pos_weight, weight_path)
        print("Weights computed and saved.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_macro_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for batch_idx, data in enumerate(loop):
            img = data['image'].to(DEVICE)
            pc = data['point_cloud'].to(DEVICE)
            targets = encode_batch_labels(data['labels'], class_to_idx).to(DEVICE)
            
            outputs = model(img, pc)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            
        # Validation
        val_metrics = validate(model, val_loader, criterion, class_to_idx, DEVICE)
        
        print(f"\nEpoch {epoch+1} Results (Baseline):")
        print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Micro F1: {val_metrics['micro_f1']:.4f}")
        
        scheduler.step(val_metrics['macro_f1'])
        
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            save_path = os.path.join(ROOT_DIR, "baseline_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best baseline model to {save_path}")

if __name__ == '__main__':
    train_baseline()
