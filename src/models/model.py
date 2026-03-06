import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=2048):
        super(ImageEncoder, self).__init__()
        # Use ResNet50 as the backbone
        resnet = models.resnet50(pretrained=True)
        # Remove the fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Output shape will be (B, 2048, 1, 1) -> (B, 2048)
        self.out_dim = out_dim

    def forward(self, x):
        # Input: (B, 3, H, W)
        x = self.features(x)
        x = torch.flatten(x, 1) # (B, 2048)
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

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleGCNLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, adj):
        # x: (B, N, in_dim) - Node features
        # adj: (B, N, N) - Adjacency matrix (normalized preferably)
        
        # Message Passing: AX
        out = torch.bmm(adj, x) # (B, N, in_dim)
        
        # Update: W(AX)
        out = self.fc(out)
        out = self.relu(out)
        return out

class SpatialReasoningModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super(SpatialReasoningModule, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(SimpleGCNLayer(in_dim, hidden_dim))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))
        # Output layer
        if num_layers > 1:
            self.layers.append(SimpleGCNLayer(hidden_dim, out_dim))
            
    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x

class SceneUnderstandingModel(nn.Module):
    def __init__(self, num_classes=37): # 37 classes in SUNRGBD
        super(SceneUnderstandingModel, self).__init__()
        
        # 1. Dual-Path Feature Extraction
        self.image_encoder = ImageEncoder(out_dim=2048)
        self.point_encoder = PointNetEncoder(in_channel=3, out_dim=1024)
        
        # 2. Cross-Modal Fusion
        # Fusion Dim = 2048 + 1024 = 3072
        self.fusion_dim = 2048 + 1024
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512) # Reduce dimension for GCN
        )
        
        # 3. Spatial Reasoning (Simulated Object Nodes)
        # Since we don't have object proposals, we simulate "N objects" or just use global feature
        # For demonstration, let's assume we maintain K "latent object queries" or nodes.
        # But simpler: Just use the global fused feature for classification directly if no objects.
        # However, the user asked for "Spatial Reasoning". 
        # Strategy: Treat the scene as a single node in a graph (trivial) or project to K nodes.
        # Let's project global feature to K latent nodes to simulate "attention" to parts.
        self.k_nodes = 8 # Number of latent nodes to reason about
        self.latent_proj = nn.Linear(512, 512 * self.k_nodes) 
        
        self.gcn = SpatialReasoningModule(in_dim=512, hidden_dim=512, out_dim=512, num_layers=2)
        
        # 4. Result Output
        self.classifier = nn.Sequential(
            nn.Linear(512 * self.k_nodes, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, pc):
        batch_size = img.size(0)
        
        # --- 1. Encode ---
        img_feat = self.image_encoder(img) # (B, 2048)
        pc_feat = self.point_encoder(pc)   # (B, 1024)
        
        # --- 2. Fuse ---
        fused = torch.cat([img_feat, pc_feat], dim=1) # (B, 3072)
        fused = self.fusion_mlp(fused) # (B, 512)
        
        # --- 3. Spatial Reasoning ---
        # Project global fused feature into K latent nodes to simulate "parts" of the scene
        nodes = self.latent_proj(fused).view(batch_size, self.k_nodes, 512) # (B, K, 512)
        
        # Construct fully connected graph (Self-Attention like adjacency)
        # Adj: (B, K, K)
        # Simple: All-to-all connection
        adj = torch.ones(batch_size, self.k_nodes, self.k_nodes).to(img.device) / self.k_nodes
        
        nodes_out = self.gcn(nodes, adj) # (B, K, 512)
        
        # --- 4. Output ---
        out = nodes_out.view(batch_size, -1) # Flatten (B, K*512)
        out = self.classifier(out) # (B, NumClasses)
        
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
