import os
from skimage import io, color, feature, transform, filters, morphology
import numpy as np

# Directory containing individual hieroglyph images
hieroglyphs_folder = './patterns/'
output_folder = './extracted_regions/'  # Folder to save extracted regions

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load target image
target_image = io.imread('chufu.jpg')
target_gray = color.rgb2gray(target_image)

# Perform edge detection to estimate hieroglyph sizes
edges = filters.sobel(target_gray)
edges = morphology.dilation(edges, morphology.disk(5))  # Adjust the disk size as needed

# Estimate hieroglyph size based on the detected edges
hieroglyph_sizes = np.where(edges > edges.mean())  # Get coordinates of edges

# Determine the patch size based on the estimated hieroglyph sizes
patch_size = max(hieroglyph_sizes[0].max() - hieroglyph_sizes[0].min(),
                 hieroglyph_sizes[1].max() - hieroglyph_sizes[1].min()) + 20  # Adding extra margin

# Extract patches from the target image
extracted_regions_count = 0  # Counter for naming extracted regions
for y in range(0, target_gray.shape[0], patch_size):
    for x in range(0, target_gray.shape[1], patch_size):
        patch = target_gray[y:y + patch_size, x:x + patch_size]
        if patch.shape == (patch_size, patch_size):  # Ensure patch size consistency
            extracted_regions_count += 1
            io.imsave(f"{output_folder}/extracted_region_{extracted_regions_count}.png", patch)

print(f"{extracted_regions_count} regions extracted and saved in {output_folder}")
