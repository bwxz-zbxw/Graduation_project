import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms

from src.data.sunrgbd_dataset import SUNRGBDDataset
from src.models.model import SceneUnderstandingModel
from train import val_collate_fn as my_collate_fn

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, "best_model.pth")
BATCH_SIZE = 1 # Analyze one by one

# --- Helper to denormalize image for plotting ---
def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# --- Hook mechanism to capture attention weights ---
attention_weights = []

def attention_hook(module, input, output):
    # output is (attn_output, attn_weights)
    # attn_weights shape: (Batch, TargetLen, SourceLen) = (B, 50, 50)
    # We detach and move to CPU to save memory
    attention_weights.append(output[1].detach().cpu())

def visualize_attention():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data (Validation Set)
    print("Loading Dataset...")
    full_dataset = SUNRGBDDataset(root_dir=ROOT_DIR)
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=my_collate_fn)
    
    # 2. Load Model
    print("Loading Model...")
    classes = [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
        'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 
        'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 
        'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 
        'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'
    ]
    model = SceneUnderstandingModel(num_classes=len(classes), use_transformer=True).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    else:
        print("Error: Model weights not found. Please train the model first.")
        return

    model.eval()

    # 3. Register Hook on the LAST Transformer Layer
    # Structure: model.transformer.layers (ModuleList) -> TransformerEncoderLayer -> self_attn (MultiheadAttention)
    # specific layer index: -1 (last layer)
    hook_handle = model.transformer.layers[-1].self_attn.register_forward_hook(attention_hook)
    
    print("Capturing Attention Maps...")
    
    # 4. Run Inference on a few samples
    num_samples = 3
    iterator = iter(loader)
    
    for i in range(num_samples):
        try:
            data = next(iterator)
        except StopIteration:
            break
            
        img = data['image'].to(DEVICE)     # (1, 3, 224, 224)
        pc = data['point_cloud'].to(DEVICE) # (1, 2048, 3)
        image_path = data['paths'][0]
        
        # Clear previous weights
        attention_weights.clear()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img, pc)
            
        # Retrieve captured weights
        # attn_map shape: (1, 50, 50) -> We take the first item -> (50, 50)
        attn_map = attention_weights[0][0].numpy() 
        
        # --- Visualization Logic ---
        # Matrix Layout: 50x50
        # Indices 0-48: Image Tokens (7x7 grid flattened)
        # Index 49: Point Cloud Token (Global)
        
        # We want to see: What did the "Point Cloud Token" look at in the Image?
        # This represents: "Given the global 3D geometry, which parts of the 2D image are most relevant?"
        # Row 49 (Point Cloud) -> Columns 0-48 (Image features)
        pc_to_img_attn = attn_map[49, 0:49] # Shape (49,)
        
        # Reshape to 7x7 grid
        grid_attn = pc_to_img_attn.reshape(7, 7)
        
        # Resize to image size (224x224) for overlay
        # Normalize to 0-1 for heatmap
        grid_attn = grid_attn - grid_attn.min()
        grid_attn = grid_attn / (grid_attn.max() + 1e-8)
        
        heatmap = cv2.resize(grid_attn, (224, 224), interpolation=cv2.INTER_NEAREST)
        # Smooth it a bit for better look
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Prepare original image
        orig_img = denormalize(img[0]) # (224, 224, 3) float 0-1
        orig_img_uint8 = np.uint8(255 * orig_img)
        orig_img_uint8 = cv2.cvtColor(orig_img_uint8, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
        
        # Overlay
        overlay = cv2.addWeighted(orig_img_uint8, 0.6, heatmap, 0.4, 0)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Original
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(orig_img_uint8, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        # Subplot 2: Attention Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(grid_attn, cmap='jet')
        plt.title("Transformer Attention (7x7 Grid)")
        plt.axis('off')

        # Subplot 3: Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Spatial Reasoning Focus\n(PC Token -> Image)")
        plt.axis('off')
        
        output_filename = f"attention_vis_sample_{i+1}.png"
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150)
        plt.close()
        
        print(f"Saved visualization to {output_filename}")

    # Remove hook
    hook_handle.remove()
    print("Done. Please check the generated .png files.")

if __name__ == "__main__":
    visualize_attention()
