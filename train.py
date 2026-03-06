import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.sunrgbd_dataset import SUNRGBDDataset
from src.models.model import SceneUnderstandingModel

# --- Configuration ---
BATCH_SIZE = 32 # Increased from 8 to saturate GPU
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use current directory as root, don't hardcode full path
ROOT_DIR = os.getcwd() 

# --- Transforms ---
# Simple transform for images: Resize to 224x224 and Normalize
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def my_collate_fn(batch):
    # Batch is a list of dicts from dataset
    # Resize images here to ensure stacking works
    
    images = []
    pcs = []
    labels = []
    paths = []
    
    resize = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    for item in batch:
        # Image
        img = item['image'] # Tensor (C, H, W) range 0-255
        
        # Normalize to 0-1 and then apply standardization
        img = img / 255.0 
        img = resize(img)
        img = normalize(img)
        
        images.append(img)
        
        # Point Cloud
        pcs.append(item['point_cloud'])
        
        # Labels
        labels.append(item['labels'])

        # Path
        paths.append(item.get('image_path', ''))
        
    try:
        images = torch.stack(images)
    except RuntimeError:
        print("Error stacking images. Check sizes.")
        print([img.shape for img in images])
        raise
        
    pcs = torch.stack(pcs)
    
    return {'image': images, 'point_cloud': pcs, 'labels': labels, 'paths': paths}

def validate(model, loader, criterion, class_to_idx, device):
    model.eval()
    total_loss = 0
    encoded_encoder = lambda batch: encode_batch_labels(batch, class_to_idx)
    
    with torch.no_grad():
        for data in loader:
            img = data['image'].to(device)
            pc = data['point_cloud'].to(device)
            targets = encoded_encoder(data['labels']).to(device)
            
            outputs = model(img, pc)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

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
    
    # Split train/val (Using subset for faster demo if needed)
    # Using 80/20 split
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Determine number of workers based on OS
    # Use more workers on Linux (e.g. 8) to speed up data loading (heavy point cloud processing)
    # On Windows, keep it low (0) to avoid multiprocessing spawn issues
    num_workers = 8 if os.name == 'posix' else 0 
    
    # num_workers=0 is safer on Windows, but use more on Linux
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=my_collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=my_collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Model
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
    
    print(f"Initializing Model with {num_classes} classes...")
    model = SceneUnderstandingModel(num_classes=num_classes).to(DEVICE)
    
    # 3. Loss & Optimizer
    # Multi-label classification -> BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    print("Starting Training...")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for batch_idx, data in enumerate(loop):
            img = data['image'].to(DEVICE) # (B, 3, 224, 224)
            pc = data['point_cloud'].to(DEVICE) # (B, 2048, 3)
            
            # Prepare targets
            targets = encode_batch_labels(data['labels'], class_to_idx).to(DEVICE)
            
            # Forward
            try:
                outputs = model(img, pc) # (B, 37)
            except RuntimeError as e:
                print(f"\nRuntime Error in Forward Pass: {e}")
                print(f"Image shape: {img.shape}, PC shape: {pc.shape}")
                break
            
            # Loss
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            
        # Validation
        val_loss = validate(model, val_loader, criterion, class_to_idx, DEVICE)
        print(f"\nEpoch {epoch+1} Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(ROOT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == '__main__':
    train()
