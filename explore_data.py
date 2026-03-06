import scipy.io as sio
import os

# Path to the metadata file
mat_path = r"c:\Users\ASUS\Desktop\Graduation_project\SUNRGBDtoolbox\Metadata\SUNRGBDMeta.mat"

def explore_sunrgbd_meta():
    if not os.path.exists(mat_path):
        print(f"Error: File not found at {mat_path}")
        return

    print(f"Loading {mat_path}...")
    try:
        data = sio.loadmat(mat_path)
        print("Keys in MAT file:", list(data.keys()))
        
        if 'SUNRGBDMeta' in data:
            meta = data['SUNRGBDMeta']
            print(f"SUNRGBDMeta shape: {meta.shape}")
            
            # Access the first sample (it's usually a structured array)
            # data['SUNRGBDMeta'] is likely (1, N)
            sample_idx = 0
            if meta.shape[1] > 0:
                sample = meta[0, sample_idx]
                print(f"\n--- Fields available in sample {sample_idx} ---")
                print(sample.dtype.names)
                
                # Check for image paths
                if 'rgbpath' in sample.dtype.names:
                    rgb_path = sample['rgbpath'][0] if len(sample['rgbpath']) > 0 else "N/A"
                    print(f"\nRGB Path (Raw): {rgb_path}")
                
                if 'depthpath' in sample.dtype.names:
                    depth_path = sample['depthpath'][0] if len(sample['depthpath']) > 0 else "N/A"
                    print(f"Depth Path (Raw): {depth_path}")

                # Check specifically for data locations relative to our workspace
                # The paths in the MAT file might be absolute paths from the original dataset authors' machine
                # We need to figure out how to map them to the pure CWD content
    except Exception as e:
        print(f"Failed to load or parse .mat file: {e}")

if __name__ == "__main__":
    explore_sunrgbd_meta()
