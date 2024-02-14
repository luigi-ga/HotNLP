import pytorch_lightning as pl
import torch
from torch import nn
from transformers import CLIPModel, CLIPTokenizer
from torch.nn.functional import cosine_similarity
from nltk.corpus import wordnet as wn
from deep_translator import GoogleTranslator


class ClipModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze CLIP model's parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.language = 'en'

    @staticmethod
    def expand_query(text):
        expanded_queries = [text]  # include the original text
        for word in text.split():
            synsets = wn.synsets(word)
            for synset in synsets:
                # Add synonyms
                expanded_queries += [syn.replace('_', ' ') for syn in synset.lemma_names()]
                # Add hypernyms and hyponyms as needed
                # expanded_queries += [lemma.name() for hypernym in synset.hypernyms() for lemma in hypernym.lemmas()]
                # expanded_queries += [lemma.name() for hyponym in synset.hyponyms() for lemma in hyponym.lemmas()]
        return list(set(expanded_queries))  # Remove duplicates
    
    def set_language(self, language):
        self.language = language

    def forward(self, texts, images):
        # Translate text to English
        if self.language != 'en':
            texts = [GoogleTranslator(source=self.language, target='en').translate(txt) for txt in texts]

        # Expand input texts into multiple queries based on WordNet
        expanded_texts = [self.expand_query(txt) for txt in texts]  # Assume texts is a list of BATCH_SIZE text inputs

        # Initialize container for all text features
        all_text_features = []
        
        # Process each expanded text query
        for text_list in expanded_texts:
            # Tokenize and compute embeddings for each sense
            text_features_list = []
            for text in text_list:
                text_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
                # Only select the necessary inputs for the CLIP model
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items() if k in ['input_ids', 'attention_mask']}
                # Compute text features using the CLIP model
                text_features = self.clip_model.get_text_features(**text_inputs).unsqueeze(0)  # Add batch dim for aggregation
                text_features_list.append(text_features)
            
            # Concatenate and average embeddings across senses for each original text input
            text_features_stack = torch.cat(text_features_list, dim=0)
            avg_text_features = torch.mean(text_features_stack, dim=0, keepdim=True)
            all_text_features.append(avg_text_features)
        
        # Concatenate all text features to match the batch size
        all_text_features = torch.cat(all_text_features, dim=0)
        
        # Process images as in your original model
        images = torch.stack(images, dim=1)
        images = images.view(-1, *images.size()[2:])
        image_features = self.clip_model.get_image_features(images)
        batch_size = len(texts)
        image_features = image_features.view(batch_size, -1, image_features.size(-1))

        return all_text_features, image_features
    
    def test_step(self, batch, batch_idx):
        text, images, gold_label_index = batch
        # text_features and image_features are already prepared for comparison
        text_features, image_features = self.forward(text, images)
        
        # Compute similarity scores between averaged text features and image features
        scores = cosine_similarity(text_features, image_features, dim=2)
        
        # Calculate H@1 (Hit at 1)
        top_pred = scores.argmax(dim=1)  # Find the index of the highest scoring image
        h_at_1 = (top_pred == gold_label_index).float().mean()  # Calculate the accuracy

        # Calculate MRR (Mean Reciprocal Rank)
        rank = (scores.argsort(descending=True).argsort(dim=1) == gold_label_index.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        mrr = (1. / rank.float()).mean()

        # Log metrics
        self.log('test_h_at_1', h_at_1, on_epoch=True)
        self.log('test_mrr', mrr, on_epoch=True)

        return {'h_at_1': h_at_1, 'mrr': mrr}



