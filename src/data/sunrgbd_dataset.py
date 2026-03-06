import os
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset

class SUNRGBDDataset(Dataset):
    def __init__(self, root_dir, split='train', split_file=None):
        """
        Args:
            root_dir (str): Path to the root of the project (where SUNRGBDtoolbox and SUNRGBD folders are).
            split (str): 'train' or 'test'.
        """
        self.root_dir = root_dir
        self.split = split
        
        # Metadata path
        self.meta_path = os.path.join(root_dir, 'SUNRGBDtoolbox', 'Metadata', 'SUNRGBDMeta.mat')
        
        # Load metadata
        print(f"Loading metadata from {self.meta_path}...")
        try:
            self.all_meta = sio.loadmat(self.meta_path, squeeze_me=True, struct_as_record=False)['SUNRGBDMeta']
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {self.meta_path}")
            
        # Split data (Simple split for now, typically there is a specific split file)
        # Using all data for demonstration if no split file provided
        self.indices = list(range(len(self.all_meta)))
        # In a real scenario, you would filter self.indices based on 'split'
        
        print(f"Loaded {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        meta_idx = self.indices[idx]
        sample = self.all_meta[meta_idx]
        
        # 1. Resolve Paths
        # The paths in mat file are like: /n/fs/sun3d/data/SUNRGBD/kv1/...
        # We need to map them to: self.root_dir/SUNRGBD/kv1/...
        
        folder_path = str(sample.sequenceName) # e.g. /n/fs/sun3d/data/SUNRGBD/kv1/b3dodata/img_0063
        
        # Find where "SUNRGBD" starts in the path
        rel_path_start = folder_path.find('SUNRGBD')
        if rel_path_start == -1:
            # Fallback or error handling
            rel_path = folder_path
        else:
            rel_path = folder_path[rel_path_start:] # SUNRGBD/kv1/...
            
        dataset_root = os.path.join(self.root_dir, rel_path)
        
        # Construct actual file paths
        rgb_name = sample.rgbname
        depth_name = sample.depthname
        
        rgb_path = os.path.join(dataset_root, 'image', rgb_name)
        depth_path = os.path.join(dataset_root, 'depth', depth_name)
        
        # 2. Load RGB Image
        image = Image.open(rgb_path).convert('RGB')
        image = np.array(image)
        
        # 3. Load Depth Image & Convert to Point Cloud
        # SUNRGBD specific depth loading (bitwise operation)
        depth = Image.open(depth_path)
        depth = np.array(depth).astype(np.uint16)
        
        # Apply bitwise shift as per SUNRGBD SDK
        depth = (depth >> 3) | (depth << 13)
        depth = depth.astype(np.float32) / 1000.0  # Convert to meters
        
        # Filter invalid depth
        depth[depth > 8.0] = 0 # Max depth 8m
        
        # Generate Point Cloud
        # We need intrinsics K and extrinsics
        K = sample.K # 3x3 Intrinsic matrix
        
        # Optimization: Downsample depth map before back-projection
        # This significantly reduces the number of points to process (CPU intensive part)
        # Stride of 4 reduces points by 16x (e.g. 300k -> 19k), still enough to sample 2048
        stride = 4
        depth = depth[::stride, ::stride]
        
        # Create meshgrid for pixel coordinates
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(0, cols*stride, stride), np.arange(0, rows*stride, stride))
        
        # Back-project to 3D
        # z = depth
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        
        valid_mask = depth > 0
        z = depth[valid_mask]
        x = (c[valid_mask] - cx) * z / fx
        y = (r[valid_mask] - cy) * z / fy
        
        # Stack to (N, 3)
        xyz = np.stack([x, y, z], axis=1)
        
        # Apply Rtilt (Tilt correction) if available to align with gravity
        # This aligns the floor to be horizontal
        if hasattr(sample, 'Rtilt'):
            Rtilt = sample.Rtilt
            # Rtilt is typically 3x3
            xyz = np.dot(xyz, Rtilt.T)
            
        # Swap axes to be consistent (e.g. Y-up or Z-up depending on convention)
        # SUNRGBD is typically: X-right, Y-down (image), Z-forward
        # After Rtilt: X-right, Z-forward (horizontal), Y-down (vertical)?
        # For now we keep as is.
        
        # 4. Sampling Points (e.g. 2048 points)
        num_points = 2048
        if xyz.shape[0] > num_points:
            choices = np.random.choice(xyz.shape[0], num_points, replace=False)
            xyz = xyz[choices]
        elif xyz.shape[0] > 0:
            choices = np.random.choice(xyz.shape[0], num_points, replace=True)
            xyz = xyz[choices]
        else:
            xyz = np.zeros((num_points, 3)) # Dummy if empty
            
        # 5. Load Bounding Boxes (Labels)
        # sample.groundtruth3DBB is a structured array
        bboxes = []
        labels = []
        
        if hasattr(sample, 'groundtruth3DBB'):
            # The structure of groundtruth3DBB is an array of structs
            gts = sample.groundtruth3DBB
            if isinstance(gts, np.ndarray) and gts.size > 0:
                # Handle single element vs array
                if gts.ndim == 0:
                    gts = np.array([gts])
                
                for gt in gts:
                    # Depending on how scipy loaded it, access fields
                    # Typical fields: basis(3x3), coeffs(3), centroid(3), classname, orientation
                    # Simplify for now: just get centroid and size
                    try:
                        centroid = gt.centroid
                        coeffs = gt.coeffs # Typically [l, w, h] or half-sizes
                        classname = gt.classname
                        bboxes.append(np.concatenate([centroid, coeffs]))
                        labels.append(classname)
                    except:
                        pass
        
        # Convert to Tensors
        image_tensor = torch.from_numpy(image).float().permute(2,0,1) # C,H,W
        pc_tensor = torch.from_numpy(xyz).float()
        
        return {
            'image': image_tensor,
            'point_cloud': pc_tensor,
            'image_path': rgb_path,
            'labels': labels
        }

if __name__ == '__main__':
    # Test block
    root = r'C:\Users\ASUS\Desktop\Graduation_project'
    dataset = SUNRGBDDataset(root)
    print("Dataset created.")
    
    # Try one sample
    try:
        data = dataset[0]
        print("Sample 0 Loaded:")
        print(f"Image shape: {data['image'].shape}")
        print(f"Point Cloud shape: {data['point_cloud'].shape}")
        print(f"Path: {data['image_path']}")
        print(f"Labels: {data['labels']}")
    except Exception as e:
        print(f"Error loading sample: {e}")
