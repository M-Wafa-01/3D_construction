import numpy as np
from mayavi import mlab
import tifffile as tiff

def load_tif_sequence(path, num_images):
    images = [tiff.imread(f"{path}Z_Fish_{i:04d}.tif") for i in range(num_images)]
    volume = np.stack(images, axis=-1)
    return volume

def visualize_with_mayavi(volume):
    mlab.pipeline.volume(mlab.pipeline.scalar_field(volume))
    mlab.show()

def main():
    path = "C:/Users/mmabo/Downloads/Image_reconstruction/"
    num_images = 113  # Adjust based on your data
    volume = load_tif_sequence(path, num_images)
    visualize_with_mayavi(volume)

if __name__ == "__main__":
    main()
