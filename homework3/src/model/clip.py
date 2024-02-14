import pytorch_lightning as pl
import torch
from torch import nn
from transformers import CLIPModel, CLIPTokenizer
from torch.nn.functional import cosine_similarity
from lavis.models import load_model_and_preprocess


class ClipModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze CLIP model's parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, text, images):
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


