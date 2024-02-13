import pytorch_lightning as pl
import torch
from transformers import CLIPModel, CLIPTokenizer
from torch.nn.functional import cosine_similarity
from deep_translator import GoogleTranslator
import spacy
import pandas as pd
import os
from spacy.lang.en.stop_words import STOP_WORDS  # Ensure you have this import if not already

# Set Transformers logging to ERROR only, which hides warnings and informational messages
from transformers import logging
logging.set_verbosity_error()


class ClipModel(pl.LightningModule):
    def __init__(self, gpt_root, language='en'):
        super().__init__()
        self.gpt_root = gpt_root
        self.language = language

        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze CLIP model's parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Load the spaCy model
        self.nlp = spacy.load("en_core_web_md")

        # Open dataframes
        self.aspect_df = pd.read_csv(os.path.join(self.gpt_root, f'gpt_{self.language}_desc.csv'))

    def set_language(self, language):
        self.language = language
        self.aspect_df = pd.read_csv(os.path.join(self.gpt_root, f'gpt_{self.language}_desc.csv'))    

    def forward(self, text, images):
        # Translate text to English and expand it with GPT-Neo
        text = [self.translate_augment_and_filter(txt) for txt in text]

        # Tokenize text inputs and send them to the device
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Process all images and compute image features
        # Assuming images is a list of tensors with shape [batch_size, 10, C, H, W]
        # We want to convert it to [batch_size * 10, C, H, W] to pass through the model
        images = torch.stack(images, dim=1)
        # Reshape images for processing: [batch_size, 10, C, H, W] -> [batch_size * 10, C, H, W]
        images = images.view(-1, *images.size()[2:])
        # Get image features from CLIP
        image_features = self.clip_model.get_image_features(images)

        # Reshape image_features back to [batch_size, 10, feature_size]
        batch_size = text_inputs['input_ids'].size(0)
        image_features = image_features.view(batch_size, -1, image_features.size(-1))

        # Compute text features from CLIP
        text_features = self.clip_model.get_text_features(**text_inputs).unsqueeze(1)

        return text_features, image_features
    
    def test_step(self, batch, batch_idx):
        text, images, gold_label_index = batch
        text_features, image_features = self.forward(text, images)
        scores = cosine_similarity(text_features, image_features, dim=2)

        # Calculate H@1
        top_pred = scores.argmax(dim=1)
        h_at_1 = (top_pred == gold_label_index).float().mean()

        # Calculate MRR
        rank = (scores.argsort(descending=True).argsort(dim=1) == gold_label_index.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        mrr = (1. / rank.float()).mean()

        # Log metrics
        self.log('test_h_at_1', h_at_1, on_epoch=True)
        self.log('test_mrr', mrr, on_epoch=True)

        return {'h_at_1': h_at_1, 'mrr': mrr}
    
    def translate_augment_and_filter(self, text, similarity_threshold=0.5):
        try:
            # Extract aspect
            aspect = self.aspect_df.loc[self.aspect_df['full_phrase'] == text, 'aspect'].iloc[0]
            translated_gpt = aspect if self.language == 'en' else GoogleTranslator(source='auto', target='en').translate(aspect)
        except IndexError:
            # Print the text variable when IndexError occurs
            print("IndexError occurred with text:", text)
        # Extract aspect
        aspect = self.aspect_df.loc[self.aspect_df['full_phrase'] == text, 'aspect'].iloc[0]
        translated_gpt = aspect if self.language == 'en' else GoogleTranslator(source='auto', target='en').translate(aspect)

        # Translate text to English if it's not already in English
        translated_text = text if self.language == 'en' else GoogleTranslator(source='auto', target='en').translate(text)

        # Process texts to obtain tokens
        doc_original = self.nlp(translated_text)
        doc_expanded = self.nlp(translated_gpt)
        
        # Filter tokens based on similarity, not being a stop word, and not being punctuation
        relevant_tokens = [
            token.text for token in doc_expanded 
            if any(token.similarity(orig_token) >= similarity_threshold for orig_token in doc_original if orig_token.has_vector and token.has_vector) 
            and not token.is_stop 
            and not token.is_punct
        ]
        
        # Combine original text with relevant tokens from the expansion
        optimized_expansion = translated_text + ' ' + ' '.join(relevant_tokens)
        
        return optimized_expansion
    