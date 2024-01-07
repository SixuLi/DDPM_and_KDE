import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

sns.set_style("darkgrid")
sns.set_context('poster')


from PIL import Image


def create_image_collage(images, collage_dim=(8, 3), image_size=(100, 100)):
    """
    Create a collage of images.

    Parameters:
    - images (list of PIL.Image): List of image objects to include in the collage.
    - collage_dim (tuple): Dimensions (width, height) of the collage in terms of number of images.
    - image_size (tuple): Size (width, height) of each image in the collage.

    Returns:
    - PIL.Image: The final collage image.
    """
    # Calculate the size of the collage
    collage_width = collage_dim[0] * image_size[0]
    collage_height = collage_dim[1] * image_size[1]

    # Create a new blank image for the collage
    collage = Image.new('RGB', (collage_width, collage_height))

    # Paste each image into the collage
    for index, img in enumerate(images):
        # Resize image if it's not the correct size
        if img.size != image_size:
            img = img.resize(image_size, Image.ANTIALIAS)

        # Calculate the position where the current image should be pasted in the collage
        x_position = (index % collage_dim[0]) * image_size[0]
        y_position = (index // collage_dim[0]) * image_size[1]

        # Paste the image
        collage.paste(img, (x_position, y_position))

    return collage


# Load the images into a list
# image_files = ['./results/KDE_generation_cifar10/kde_sampling_original_cifar10_{}.png'.format(i) for i in range(0, 8)] + ['./results/KDE_generation_cifar10/kde_sampling_sample_cifar10_{}.png'.format(i) for i in range(0, 8)] + \
#                 ['./results/fid-tmp-optim-early-stop-3/000000/00002{}.png'.format(i) for i in range(0, 8)] # Adjust the range and file names as necessary
image_files = ['./results/fid-tmp-optim-early-stop-0/000000/00002{}.png'.format(i) for i in range(0, 8)] + \
              ['./results/fid-tmp-optim-early-stop-3/000000/00002{}.png'.format(i) for i in range(0, 8)] + \
              ['./results/fid-tmp-optim-early-stop-5/000000/00002{}.png'.format(i) for i in range(0, 8)]
images = [Image.open(image_file) for image_file in image_files]

# Generate the collage
collage = create_image_collage(images)

# Save the collage
output_path = './results/collected_visualizations_different_time_step.png'
collage.save(output_path)

# Display the collage
collage.show()


