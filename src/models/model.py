import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=2048):
        super(ImageEncoder, self).__init__()
        # Use ResNet50 as the backbone
        # Modified to keep spatial dimensions (7x7) by removing AvgPool and FC
        
        # Configure weight download path
        torch.hub.set_dir(os.path.join(os.getcwd(), 'models', 'weights'))
        
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        
        # Remove AvgPool (last layer) and FC (second to last is AvgPool in children list sometimes, let's be safe)
        # Standard ResNet children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # We want up to layer4 -> all except last 2
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.out_dim = out_dim

    def forward(self, x):
        # Input: (B, 3, H, W) -> (B, 2048, H/32, W/32)
        # For 224x224 input, output is (B, 2048, 7, 7)
        x = self.features(x)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, in_channel=3, out_dim=1024):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # Input: (B, N, 3) -> Transpose to (B, 3, N)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # Global Max Pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class SceneUnderstandingModel(nn.Module):
    def __init__(self, num_classes=37): # 37 classes in SUNRGBD
        super(SceneUnderstandingModel, self).__init__()
        
        # 1. Feature Extraction
        self.image_encoder = ImageEncoder(out_dim=2048) # Outputs (B, 2048, 7, 7)
        self.point_encoder = PointNetEncoder(in_channel=3, out_dim=1024) # Outputs (B, 1024)
        
        # 2. Projection layers to common dimension for Transformer
        self.hidden_dim = 512
        self.img_proj = nn.Conv2d(2048, self.hidden_dim, kernel_size=1)
        self.pc_proj = nn.Linear(1024, self.hidden_dim)
        
        # 3. Spatial Reasoning (Transformer Encoder)
        # We treat 7x7 image regions + 1 global PC feature as a sequence of 50 tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. Result Output
        # We classify based on the mean of all tokens (Global Context)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, pc):
        batch_size = img.size(0)
        
        # --- 1. Encode ---
        img_feat = self.image_encoder(img) # (B, 2048, 7, 7)
        pc_feat = self.point_encoder(pc)   # (B, 1024)
        
        # --- 2. Prepare Sequence ---
        # Image Tokens: (B, 512, 7, 7) -> (B, 512, 49) -> (B, 49, 512)
        img_tokens = self.img_proj(img_feat).flatten(2).permute(0, 2, 1) 
        
        # Point Cloud Token: (B, 1024) -> (B, 512) -> (B, 1, 512)
        pc_token = self.pc_proj(pc_feat).unsqueeze(1)
        
        # Concatenate: [Image_Tokens, PC_Token] -> (B, 50, 512)
        tokens = torch.cat([img_tokens, pc_token], dim=1)
        
        # --- 3. Spatial Reasoning (Transformer) ---
        # Tokens interact with each other via Self-Attention
        attended_tokens = self.transformer(tokens) # (B, 50, 512)
        
        # --- 4. Output ---
        # Global Average Pooling over all tokens to get scene descriptor
        scene_feat = attended_tokens.mean(dim=1) # (B, 512)
        
        out = self.classifier(scene_feat) # (B, NumClasses)
        
        return out

if __name__ == '__main__':
    # Test Block
    model = SceneUnderstandingModel()
    
    # Dummy Input
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_pc = torch.randn(2, 2048, 3) # (B, N, 3)
    
    print("Testing Model Forward Pass...")
    try:
        output = model(dummy_img, dummy_pc)
        print(f"Output Shape: {output.shape}") # Should be (2, 37)
        print("Model Verified.")
    except Exception as e:
        print(f"Error: {e}")
