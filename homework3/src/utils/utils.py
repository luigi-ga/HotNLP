import os
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.pyplot as plt

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
                img_resized = img.resize(target_size)
                
                # Save the image, overwriting the original
                img_resized.save(filepath)


def display_samples(dataset, num_samples=3):
    # Loop through the dataset
    for i, sample in enumerate(dataset):
        # Display num_samples samples
        if i < num_samples:
            fig, axs = plt.subplots(1, len(sample[1]), figsize=(10,1))
            for i, image in enumerate(sample[1]):
                img = image.numpy().transpose((1, 2, 0))
                axs[i].imshow(img, alpha = 1 if i==sample[2].item() else 1/3)
                axs[i].axis('off')
            # Adding a central title to the figure
            fig.suptitle(f"{sample[0]}", fontsize=10)
            plt.show()
        else:
            break