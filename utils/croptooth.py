from PIL import Image
import multiprocessing
import csv
from tqdm import tqdm
import ast
import numpy as np

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def crop_boundingbox(image_path, segmentation, save_path):

    box_min, box_max = bounding_box(segmentation)
    # Open the image
    image = Image.open(image_path)
    image_array = np.array(image)

    cropped_image = image_array[box_min[1]:box_max[1], box_min[0]:box_max[0]]

    # Convert array back to image and save the result
    cropped_tooth_image = Image.fromarray(cropped_image)
    resized_image = cropped_tooth_image.resize((224, 224))

    # Save the resized cropped image
    resized_image.save(save_path)

    return save_path

if __name__ == "__main__" :
        
    # Replace 'input.csv' with the path to your actual CSV file
    csv_file_path = '../../healthcare_baseline/utils/train_input.csv'

    num_process = 4
    # processes = ["p1","p2","p3","p4"]
    pool = multiprocessing.Pool(processes = num_process)

    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # Create a CSV reader
        csv_reader = csv.DictReader(csv_file)
        
        # Loop through the rows in the CSV file
        for row in tqdm(csv_reader, desc="Processing Images"):
            # if row['path'].split('_')[0] != 'front':
            #     continue
            # Print the 'decayed' column
            #TODO newname 주소 바꾸기
            newname = '../../Dataset/train_data/imagecrop/' + row['path'].replace('.png','') + '_' + row['teeth_num'] + '.png'
            segmentation_example = ast.literal_eval(row['segmentation'])
            imgpath = '../../Dataset/train_data/image/' + row['path']
            # create_masked_tooth_image(imgpath, segmentation_example, newname)
            pool.apply_async(crop_boundingbox, args=(imgpath, segmentation_example, newname))
            break
        pool.close()
        pool.join()
