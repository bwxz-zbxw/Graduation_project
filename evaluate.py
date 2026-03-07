import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.data.sunrgbd_dataset import SUNRGBDDataset
from src.models.model import SceneUnderstandingModel
from train import my_collate_fn, encode_batch_labels

# --- Configuration ---
BATCH_SIZE = 32
CONF_THRESHOLD = 0.25 # Lower threshold to improve recall for multi-label
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = os.getcwd() 

# Colors for visualization
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def get_classes():
    return [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
        'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 
        'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 
        'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 
        'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'
    ]

def evaluate_and_visualize():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print("Initializing Dataset...")
    full_dataset = SUNRGBDDataset(root_dir=ROOT_DIR)
    
    # Same split logic as train.py
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Use fewer threads for evaluation to be safe
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, # Shuffle to see random visuals
        collate_fn=my_collate_fn, 
        num_workers=4 if os.name == 'posix' else 0
    )
    
    classes = get_classes()
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # 2. Load Model
    print("Loading Best Model...")
    model = SceneUnderstandingModel(num_classes=len(classes)).to(DEVICE)
    model_path = os.path.join(ROOT_DIR, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. Metrics Calculation
    all_preds = []
    all_targets = []
    
    print("Calculating metrics on Validation Set...")
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Evaluating"):
            img = data['image'].to(DEVICE)
            pc = data['point_cloud'].to(DEVICE)
            
            # Ground Truth
            target_labels = encode_batch_labels(data['labels'], class_to_idx).numpy()
            all_targets.append(target_labels)
            
            # Prediction
            outputs = model(img, pc)
            # Sigmoid converts logits to probabilities (0-1)
            probs = torch.sigmoid(outputs)
            # Threshold using configured value
            preds = (probs > CONF_THRESHOLD).cpu().numpy().astype(int)
            all_preds.append(preds)

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    
    # Calculate Metrics
    # Micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # Macro: Calculate metrics for each label, and find their unweighted mean.
    f1_micro = f1_score(all_targets, all_preds, average='micro')
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    acc_subset = accuracy_score(all_targets, all_preds) # Exact match ratio (harsh metric)
    
    # Precision/Recall
    precision_micro = precision_score(all_targets, all_preds, average='micro')
    recall_micro = recall_score(all_targets, all_preds, average='micro')

    print("\n" + "="*30)
    print("       EVALUATION RESULTS       ")
    print("="*30)
    print(f"F1 Score (Micro):    {f1_micro:.4f} (Weighted importance)")
    print(f"F1 Score (Macro):    {f1_macro:.4f} (Average across classes)")
    print(f"Precision (Micro):   {precision_micro:.4f}")
    print(f"Recall (Micro):      {recall_micro:.4f}")
    print(f"Subset Accuracy:     {acc_subset:.4f} (All labels must match exactly)")
    print("="*30 + "\n")

    # 4. Visualization (Save to file)
    print("Generating visualizations...")
    visualize_samples(val_loader, model, classes, DEVICE, num_samples=5)

def visualise_single_prediction(img_tensor, pred_indices, target_indices, classes, save_path):
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    pred_labels = [classes[i] for i in pred_indices]
    gt_labels = [classes[i] for i in target_indices]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    
    # Text
    pred_str = ", ".join(pred_labels)
    gt_str = ", ".join(gt_labels)
    
    plt.title(f"Pred: {pred_str}\nGT: {gt_str}", fontsize=10, loc='left', wrap=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

def visualize_samples(loader, model, classes, device, num_samples=5):
    with torch.no_grad():
        # Get one batch
        data = next(iter(loader))
        imgs = data['image'].to(device)
        pcs = data['point_cloud'].to(device)
        raw_labels = data['labels'] # List of lists
        
        outputs = model(imgs, pcs)
        probs = torch.sigmoid(outputs)
        
        # Create 'results' directory
        os.makedirs("results", exist_ok=True)
        
        for i in range(min(num_samples, len(imgs))):
            # Get predictions > 0.5
            pred_indices = torch.where(probs[i] > 0.5)[0].cpu().numpy()
            
            # Get Ground Truth indices
            # Need to map raw strings back to indices for consistent comparison logic or just use strings
            gt_indices = []
            class_to_idx = {c: idx for idx, c in enumerate(classes)}
            for l in raw_labels[i]:
                if l in class_to_idx:
                    gt_indices.append(class_to_idx[l])
            
            visualise_single_prediction(
                imgs[i], 
                pred_indices, 
                gt_indices, 
                classes, 
                save_path=f"results/val_sample_{i}.png"
            )

if __name__ == '__main__':
    evaluate_and_visualize()
