import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import random

from src.data.sunrgbd_dataset import SUNRGBDDataset
from src.models.model import SceneUnderstandingModel

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, "best_model.pth")
NUM_SAMPLES = 6
CONF_THRESHOLD = 0.5

# Classes (Must match train.py)
ALL_CLASSES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 
    'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 
    'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 
    'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'
]

def load_model():
    model = SceneUnderstandingModel(num_classes=len(ALL_CLASSES))
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Error: Best model not found!")
        exit(1)
    model.to(DEVICE)
    model.eval()
    return model

def predict_single(model, image, pc):
    # Preprocess Image
    # val_transform from train.py logic
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert RGB numpy to Tensor (C,H,W)
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).byte() # Assuming image is H,W,C uint8
    img_input = transform(img_tensor).unsqueeze(0).to(DEVICE) # (1, 3, 224, 224)
    
    # Preprocess PC
    pc_input = torch.tensor(pc).unsqueeze(0).float().to(DEVICE) # (1, N, 3)
    
    with torch.no_grad():
        logits = model(img_input, pc_input)
        probs = torch.sigmoid(logits)
        
    return probs.cpu().numpy()[0]

def visualize():
    model = load_model()
    dataset = SUNRGBDDataset(root_dir=ROOT_DIR)
    
    # Pick random samples from validation set (approx last 20%)
    total_len = len(dataset)
    val_start = int(0.8 * total_len)
    val_indices = list(range(val_start, total_len))
    selected_indices = random.sample(val_indices, NUM_SAMPLES)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    print(f"Visualizing {NUM_SAMPLES} samples...")
    
    for i, idx in enumerate(selected_indices):
        data = dataset[idx]
        image = data['image'].permute(1, 2, 0).numpy().astype(np.uint8) # H,W,C
        pc = data['point_cloud'].numpy()
        gt_labels = data['labels']
        
        # Predict
        probs = predict_single(model, image, pc)
        
        # Get predicated classes
        pred_labels = []
        scores = []
        for class_idx, score in enumerate(probs):
            if score > CONF_THRESHOLD:
                pred_labels.append(ALL_CLASSES[class_idx])
                scores.append(score)
        
        # Sort predictions by score
        pred_pairs = sorted(zip(pred_labels, scores), key=lambda x: x[1], reverse=True)
        pred_str = ", ".join([f"{name}" for name, score in pred_pairs])
        if not pred_str:
            pred_str = "None"
            
        gt_str = ", ".join(list(set(gt_labels))) # Remove duplicates
        
        # Plot
        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')
        
        # Title with color coding
        # Green title if at least one correct prediction intersect, else Red ??
        # Just simple text for now
        title = f"GT: {gt_str}\nPred: {pred_str}"
        
        # Wrap text if too long
        import textwrap
        title = "\n".join(textwrap.wrap(title, width=40))
        
        ax.set_title(title, fontsize=10, loc='left')
        
    plt.tight_layout()
    plt.savefig('visualization_results.png')
    print("Saved visualization to visualization_results.png")

if __name__ == '__main__':
    visualize()
