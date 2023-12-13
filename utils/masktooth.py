from PIL import Image, ImageDraw
import numpy as np
import ast
import tqdm

def create_masked_tooth_image(image_path, segmentation, save_path):
    """
    Create a masked image for a given tooth segmentation.
    
    Parameters:
    image_path (str): The path to the original image.
    segmentation (list): The segmentation data for the tooth.
    save_path (str): The path to save the masked image.
    """
    # Open the image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Flatten the segmentation data into a tuple of coordinates
    segmentation_tuple = tuple([coord for point in segmentation for coord in point])

    # Create mask image
    mask_image = Image.new('L', (image_array.shape[1], image_array.shape[0]), 0)
    ImageDraw.Draw(mask_image).polygon(segmentation_tuple, outline=1, fill=1)
    mask = np.array(mask_image)

    # Apply mask to the image: keep the tooth area, blackout the rest
    image_array[mask == 0] = 0  # Blackout outside the mask

    # Convert array back to image and save the result
    masked_tooth_image = Image.fromarray(image_array)
    resized_image = masked_tooth_image.resize((224, 224))

    # Save the resized masked image
    resized_image.save(save_path)

    return save_path

import csv

# Replace 'input.csv' with the path to your actual CSV file
csv_file_path = '../../healthcare_baseline/utils/train_input.csv'

# Open the CSV file
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    # Create a CSV reader
    csv_reader = csv.DictReader(csv_file)
    
    # Loop through the rows in the CSV file
    for row in tqdm(csv_reader, desc="Processing Images"):
        # Print the 'decayed' column
        newname = '../../Dataset/train_data/imagemask/' + row['path'].replace('.png','') + '_' + row['teeth_num'] + '.png'
        segmentation_example = ast.literal_eval(row['segmentation'])
        imgpath = '../../Dataset/train_data/image/' + row['path']
        create_masked_tooth_image(imgpath, segmentation_example, newname)

# Example segmentation data


# Call the function with the image path, segmentation data, and the desired save path for the masked image
