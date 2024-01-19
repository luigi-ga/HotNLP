import os
from tqdm.notebook import tqdm
from PIL import Image

def resize_images(data_dir, target_size=(224, 224)):
    # Iterate over each file in the directory
    for filename in tqdm(os.listdir(data_dir), desc='Resizing images'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(data_dir, filename)
            
            # Open the image
            with Image.open(filepath) as img:
                # Convert RGBA to RGB
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Resize the image
                img_resized = img.resize(target_size, Image.ANTIALIAS)
                
                # Save the image, overwriting the original
                img_resized.save(filepath)