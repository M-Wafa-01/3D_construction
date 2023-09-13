import numpy as np
import torch
import os
import cv2
import tifffile as tiff
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def preprocess_image(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 80, 255, cv2.THRESH_BINARY)
    return img_thresh

def visualize_3d(volume):
    verts, faces, _, _ = marching_cubes(volume, level=0.5)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)
    plt.show()

def main():
    directory_path = "C:/Users/mmabo/Downloads/Image_reconstruction/"
    num_images = len([name for name in os.listdir(directory_path) if name.startswith("Z_Fish_") and name.endswith(".tif")])

    # Adjust the reading pattern to match your filenames
    images = [tiff.imread(f"{directory_path}Z_Fish_{i:04d}.tif") for i in range(num_images)]

    # Move list of 2D images to a single 3D numpy array and convert to float32
    volume_np = np.stack(images, axis=2).astype(np.float32)

    # Convert numpy volume to PyTorch tensor and move to GPU
    volume_torch = torch.tensor(volume_np).cuda()

    # Example of processing on GPU
    volume_torch -= volume_torch.mean()
    print(volume_torch.device)
    # Convert back to numpy for visualization
    volume_processed = volume_torch.cpu().numpy()

    print(f"Volume Shape: {volume_processed.shape}")
    print(f"Min Value: {np.min(volume_processed)}, Max Value: {np.max(volume_processed)}")

    visualize_3d(volume_processed)

if __name__ == "__main__":
    main()
