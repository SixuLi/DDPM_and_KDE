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
# Adjust the range and file names as necessary
# image_files = ['./results/fid-tmp-optim-early-stop-0/000000/00002{}.png'.format(i) for i in range(0, 8)] + \
#               ['./results/fid-tmp-optim-early-stop-3/000000/00002{}.png'.format(i) for i in range(0, 8)] + \
#               ['./results/fid-tmp-optim-early-stop-5/000000/00002{}.png'.format(i) for i in range(0, 8)]
# images = [Image.open(image_file) for image_file in image_files]
#
# # Generate the collage
# collage = create_image_collage(images)
#
# # Save the collage
# output_path = './results/collected_visualizations_different_time_step.png'
# collage.save(output_path)
#
# # Display the collage
# collage.show()

# Visualize the estimated total correlation
file_path = '../results/estimate_total_correlation_d_5/result.txt'

def extract_tc_from_txt(file_path):
    gamma_values = []
    correlation_values = []

    # Read and parse the file
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains the required data before parsing
            if 'gamma' in line and ':' in line:
                # Extracting the gamma value
                gamma_str = line.split('gamma=')[1].split(':')[0].strip()
                gamma = float(gamma_str)
                gamma_values.append(gamma)

                # Extracting the list of correlations
                correlations_str = line.split('[')[1].split(']')[0].strip()
                correlations = [float(num) for num in correlations_str.split(',')]
                correlation_values.append(correlations)

    # Display the extracted lists
    # print("Gamma values:", gamma_values)
    # print("Correlation values:", correlation_values)

    return (gamma_values, correlation_values)

gamma_values, tc_values = extract_tc_from_txt(file_path=file_path)

def visualize_estimated_tc_values(gamma_values, tc_values):
    # Calculate the mean and standard deviation for each list of correlation values
    means = [np.mean(corr) for corr in tc_values]
    std_devs = [np.std(corr) for corr in tc_values]

    # Create a line plot of the gamma values against the mean correlation values
    plt.figure(figsize=(16, 10))
    plt.plot(gamma_values, means)

    # Add a band to represent the standard deviation
    # plt.fill_between(gamma_values, np.subtract(means, std_devs), np.add(means, std_devs), alpha=0.2,
    #                  label='Standard Deviation')

    plt.xlabel(r'Value of Bandwidth $\gamma$')
    plt.ylabel('Estimated Total Correlation')
    plt.title(r'Estimated Total Correlation versus Different Bandwidth $\gamma$')
    # plt.legend()
    plt.xscale('log')  # Using logarithmic scale for x-axis to better visualize the data
    plt.grid(True, linestyle='--')
    plt.savefig('../results/estimate_total_correlation/d_5.png')
    plt.show()

visualize_estimated_tc_values(gamma_values=gamma_values, tc_values=tc_values)




