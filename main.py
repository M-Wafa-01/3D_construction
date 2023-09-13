import cv2
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes


def load_tiff_stack(path):
    """Load all slices from a multi-page TIFF."""
    return imread(path)


def preprocess_image(img):
    """Preprocess image: remove noise and enhance features."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def visualize_3d(volume, level=128):
    """Visualize the 3D volume using marching cubes and mplot3d."""
    verts, faces, _, _ = marching_cubes(volume, level, spacing=(1, 1, 1))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                    linewidth=0.2, antialiased=True)
    plt.show()


def main():
    path = "C:/Users/mmabo/Downloads/FISH2222-10.tif"

    images = load_tiff_stack(path)
    preprocessed_images = [preprocess_image(img) for img in images]
    volume = np.stack(preprocessed_images, axis=2)

    # Debugging outputs
    print(f"Volume Shape: {volume.shape}")
    print(f"Min Value: {np.min(volume)}, Max Value: {np.max(volume)}")

    # 3D visualization
    visualize_3d(volume, level=30)

if __name__ == "__main__":
    main()

