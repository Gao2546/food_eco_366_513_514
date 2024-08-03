#https://youtu.be/SQng3eIEw-k
"""

Convert coco json labels to labeled masks and copy original images to place 
them along with the masks. 

Dataset from: https://github.com/sartorius-research/LIVECell/tree/main
Note that the dataset comes with: 
Creative Commons Attribution - NonCommercial 4.0 International Public License
In summary, you are good to use it for research purposes but for commercial
use you need to investigate whether trained models using this data must also comply
with this license - it probably does apply to any derivative work so please be mindful. 

You can directly download from the source github page. Links below.

Training json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json
Validation json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json
Test json: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json
Images: Download images.zip by following the link: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip

If these links do not work, follow the instructions on their github page. 


"""

import json
import PIL.Image
import numpy as np
import skimage
import tifffile
import os
import shutil
import matplotlib.pyplot as plt
import tqdm

colors = {1:[255,0,0],2:[0,255,0],3:[0,0,255]}

def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info['height'], image_info['width'],3), dtype=np.uint8)

    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            object_color = colors[ann['category_id']]
            # Extract segmentation polygon
            for seg in ann['segmentation']:
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                mask_np[rr, cc,:] = object_color
                object_number += 1 #We are assigning each object a unique integer value (labeled mask)

    # Save the numpy array as a TIFF using tifffile library or rgb .jpg image using pyplot
    mask_path = os.path.join(output_folder, image_info['file_name'].replace('.jpg', '_mask.jpg'))
    plt.imsave(mask_path,mask_np)
    # tifffile.imsave(mask_path, mask_np)

    print(f"Saved mask for {image_info['file_name']} to {mask_path}")


def main(json_file, mask_output_folder, image_output_folder, original_image_dir):
    # Load COCO JSON annotations
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    # Ensure the output directories exist
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for img in tqdm.tqdm(images):
        # Create the masks
        create_mask(img, annotations, mask_output_folder)
        
        # Copy original images to the specified folder
        original_image_path = os.path.join(original_image_dir, img['file_name'])
    
        new_image_path = os.path.join(image_output_folder, os.path.basename(original_image_path))
        shutil.copy2(original_image_path, new_image_path)
        # print(f"Copied original image to {new_image_path}")


if __name__ == '__main__':
    original_image_dir = '/home/athip/psu/3/ecosys/food_github/food_segmentation.v5i.coco-segmentation/train'  # Where your original images are stored
    json_file = '/home/athip/psu/3/ecosys/food_github/food_segmentation.v5i.coco-segmentation/train/_annotations.coco.json'
    mask_output_folder = '/home/athip/psu/3/ecosys/food_github/data/segmentation/train/masks'  # Modify this as needed. Using val2 so my data is not overwritten
    image_output_folder = '/home/athip/psu/3/ecosys/food_github/data/segmentation/train/images'  # 
    main(json_file, mask_output_folder, image_output_folder, original_image_dir)



