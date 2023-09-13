


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.io import imread


def load_tif_sequence(directory, filename_pattern, num_images):
    """Load a sequence of TIF images from a directory into a 3D numpy array."""
    images = []

    for i in range(num_images):
        file_path = os.path.join(directory, filename_pattern.format(i))
        img = imread(file_path)
        images.append(img)

    return np.stack(images)

def maximum_intensity_projection(volume):
    """Display Maximum Intensity Projection of a 3D volume."""

    mip_x = np.max(volume, axis=2)
    mip_y = np.max(volume, axis=1)
    mip_z = np.max(volume, axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(mip_x, cmap='gray')
    axs[0].set_title('MIP in X-axis')

    axs[1].imshow(mip_y, cmap='gray')
    axs[1].set_title('MIP in Y-axis')

    axs[2].imshow(mip_z, cmap='gray')
    axs[2].set_title('MIP in Z-axis')

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Usage
volume = load_tif_sequence('C:/Users/mmabo/Downloads/Image_reconstruction/', 'Z_Fish_{:04d}.tif', 113)
maximum_intensity_projection(volume)
