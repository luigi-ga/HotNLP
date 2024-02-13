import os
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from src.model.clip_gpt import ClipModel
import pandas as pd
import torch
from tensorboard.backend.event_processing import event_accumulator

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

def generate_output_2(model, lan, dataset, output_dir='./output'):
    # Load the test data
    test_df = pd.read_table(f'data/test.data.v1.1.gold/{lan}.test.data.v1.1.txt', header=None, names=['target', 'full_phrase', 'image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9'])
    
    # Open the output file for writing predictions
    with open(os.path.join(output_dir, f'{lan}.test.preds.txt'), 'w') as preds_file:
        # Iterate over the dataset
        for full_phrase, images, gold_idx in tqdm(dataset, desc=f'Testing {lan} data'):
            # Get the text and image features
            text_features, image_features = model([full_phrase], [img.unsqueeze(0) for img in images])
            # Calculate the similarity scores
            similarity_scores = torch.nn.functional.cosine_similarity(text_features.squeeze(0).squeeze(0), image_features.squeeze(0))
            # Determine the index of the highest similarity score
            predicted_idx = torch.argmax(similarity_scores).item()
            # Retrieve the filename of the predicted image
            predicted_image_filename = test_df[test_df['full_phrase'] == full_phrase][f'image{predicted_idx}'].iloc[0]
            # Write the predicted image filename to the output file
            preds_file.write(predicted_image_filename + '\n')

def generate_output(model, lan, dataloader, output_dir='./output'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Load the test data mappings
    test_df = pd.read_table(f'data/test.data.v1.1.gold/{lan}.test.data.v1.1.txt', header=None, names=['target', 'full_phrase', 'image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9'])
    
    # Open the output file for writing predictions
    with open(os.path.join(output_dir, f'{lan}.test.preds.txt'), 'w') as preds_file:
        # Iterate over the dataloader
        for batch in tqdm(dataloader, desc=f'Testing {lan} data'):
            # Extract the full phrases and images from the batch
            text, images, _ = batch

            text_features, image_features = model(text, images)
            scores = torch.nn.functional.cosine_similarity(text_features, image_features, dim=2)

            # Find predicted image filenames
            top_pred_indices = scores.argmax(dim=1).tolist()

            for i, text in enumerate(text):
                predicted_filename = test_df[test_df['full_phrase'] == text].iloc[0][f'image{top_pred_indices[i]}']
                # Write the predicted filename to the output file
                preds_file.write(predicted_filename + '\n')

def compare_files(file_path1, file_path2):
    # Initialize counters
    total_lines = 0
    matching_lines = 0

    # Open both files and iterate over their lines simultaneously
    with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            total_lines += 1
            if line1.strip() == line2.strip():
                matching_lines += 1

    # Calculate the ratio of matching lines
    if total_lines > 0:
        ratio = matching_lines / total_lines
        print(f"Matching Lines Ratio: {ratio:.2f} ({matching_lines}/{total_lines})")
    else:
        print("No lines to compare.")

def plot_pl_logs(log_dir, metric, save_dir):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Capitalize metric name
    metric_str = ' '.join(word.capitalize() for word in metric.split('_'))

    # Find the event file in the log directory
    event_file = next(file for file in os.listdir(log_dir) if 'events.out.tfevents' in file)
    full_path = os.path.join(log_dir, event_file)

    # Load the TensorBoard event file
    ea = event_accumulator.EventAccumulator(full_path)
    ea.Reload()

    # Extracting the scalar 'loss'
    if metric in ea.scalars.Keys():
        metric_df = pd.DataFrame(ea.scalars.Items(metric))

        # Plotting
        plt.figure(figsize=(6, 4))
        plt.plot(metric_df['step'], metric_df['value'], label=f'{metric_str}')
        plt.xlabel('Steps')
        plt.ylabel(f'{metric_str}')
        plt.title(f'{metric_str} Over Time')
        plt.savefig(os.path.join(save_dir, f'{metric}.pdf'), bbox_inches='tight')
    else:
        print(f"{metric} data not found in logs")