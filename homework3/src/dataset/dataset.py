import os
import torch
from PIL import Image
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from transformers import MarianMTModel, MarianTokenizer
    

class VisualWSDDatasetCLIP(Dataset):
    def __init__(self, data_dir, data_file, gold_file, transform=None, max_samples=None, language='en'):
        """
        Args:
            data_dir (string): Directory with all the images.
            data_file (string): Path to the file with annotations.
            gold_file (string): Path to the file with gold labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.data_info = self._read_data_file(data_file[language])
        self.gold_labels = self._read_gold_file(gold_file[language])
        self.transform = transform
        self.max_samples =  max_samples
        self.language = language

        # Initialize translation model and tokenizer if needed
        #if self.language != 'en':
        #    model_name = f'Helsinki-NLP/opus-mt-{self.language}-en'
        #    self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        #    self.translation_model = MarianMTModel.from_pretrained(model_name)

        # Build data
        self.data = self.build_dataset()

    def build_dataset(self):
        # Initialize the data list
        data = []

        # Iterate over the data
        for idx in tqdm(range(len(self.data_info)), 'Building VisualWSDDataset'):
            # Break if max_samples is specifies
            if self.max_samples and idx == self.max_samples: break
            # Read images as PIL, transform, and add them to a list
            images = [os.path.join(self.data_dir, img_name) for img_name in self.data_info[idx][2:]]
            # Find the index of the gold label in the images list
            gold_label_index = torch.tensor(self.data_info[idx][2:].index(self.gold_labels[idx]))

            # Append data
            data.append([self.data_info[idx][1], images, gold_label_index])

        # Return data
        return data
    
    def translate(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = self.translation_model.generate(**inputs)
            return self.tokenizer.decode(translated[0], skip_special_tokens=True)

    def _read_data_file(self, file_path):
        data_info = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 1:  # Make sure it's not an empty line
                    data_info.append(parts)
        return data_info

    def _read_gold_file(self, file_path):
        gold_labels = []
        with open(file_path, 'r') as f:
            for line in f:
                gold_labels.append(line.strip())
        return gold_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.data[idx][0]
        images = self.data[idx][1]
        gold_label_index = self.data[idx][2]
        
        if self.transform:
            images = [self.transform(Image.open(img_path)) for img_path in images]

        # Translate the phrase if necessary
        #if self.language != 'en':
        #    label = self.translate(label)

        return label, images, gold_label_index
    

class TripletsWSDDatasetCLIP(Dataset):
    def __init__(self, dataset, transform=None):
        """
        Args:
            triplets (list): List of triplets.
        """
        self.triplets = self.build_triplets(dataset)
        self.transform = transform

    def build_triplets(self, dataset):
        triplets = []
        for i, sample in enumerate(tqdm(dataset, desc="Generating triplets")):
            label, images, gold_label_index = sample
            for i in range(len(images)):
                if i != gold_label_index.item():
                    triplets.append((label, images[gold_label_index.item()], images[i]))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pos_image = self.triplets[idx][1]
        neg_image = self.triplets[idx][2]
        
        if self.transform:
            pos_image = self.transform(Image.open(pos_image))
            neg_image = self.transform(Image.open(neg_image))

        return self.triplets[idx][0], pos_image, neg_image