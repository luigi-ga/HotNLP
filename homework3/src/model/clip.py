import pytorch_lightning as pl
import torch
from torch import nn
from transformers import CLIPModel, CLIPTokenizer
from torch.nn.functional import cosine_similarity


class ClipModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze CLIP model's parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Define projection layers
        self.vision_proj = nn.Linear(512, 128)
        self.text_proj = nn.Linear(512, 128)

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
        # Project image features to the same dimension as text features
        image_features = self.vision_proj(image_features)

        # Reshape image_features back to [batch_size, 10, feature_size]
        batch_size = text_inputs['input_ids'].size(0)
        image_features = image_features.view(batch_size, -1, image_features.size(-1))

        # Compute text features from CLIP
        text_features = self.clip_model.get_text_features(**text_inputs).unsqueeze(1)
        # Project text features to the same dimension as image features
        text_features = self.text_proj(text_features)

        return text_features, image_features
    
    def compute_margin_ranking_loss(self, scores, gold_label_index):
        # Get the score of the gold label (positive score)
        positive_scores = scores.gather(1, gold_label_index.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]

        # Create a mask to filter out positive scores from negative scores
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(1, gold_label_index.unsqueeze(1), False)
        negative_scores = scores[mask].view(scores.size(0), -1)  # Shape: [batch_size, num_negatives]

        # Replicate positive_scores to match negative_scores shape
        positive_scores = positive_scores.unsqueeze(1).expand_as(negative_scores)  # Shape: [batch_size, num_negatives]

        # The target tensor indicating that positive score should be higher than negative scores
        target = torch.ones_like(positive_scores)  # Shape: [batch_size, num_negatives]

        # Use MarginRankingLoss across all negative samples
        loss_fn = nn.MarginRankingLoss(margin=0.1)
        loss = loss_fn(positive_scores, negative_scores, target)

        return loss.mean()

    def training_step(self, batch, batch_idx):
        text, images, gold_label_index = batch
        # Pass inputs to the model
        text_features, image_features = self.forward(text, images)   
        # Calculate cosine similarity scores between text and all images
        scores = cosine_similarity(text_features, image_features, dim=2)
        # Compute the loss
        loss = self.compute_margin_ranking_loss(scores, gold_label_index)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        # Accuracy is the number of correct predictions divided by the number of predictions
        accuracy = (scores.argmax(dim=1) == gold_label_index).float().mean() 
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)

        return {'loss': loss, 'accuracy': accuracy}
        
    def validation_step(self, batch, batch_idx):
        text, images, gold_label_index = batch
        # Pass inputs to the model
        text_features, image_features = self.forward(text, images)
        # Calculate cosine similarity scores between text and all images
        scores = cosine_similarity(text_features, image_features, dim=2)
        # Compute the loss
        loss = self.compute_margin_ranking_loss(scores, gold_label_index)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        
        # Accuracy is the number of correct predictions divided by the number of predictions
        accuracy = (scores.argmax(dim=1) == gold_label_index).float().mean() 
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True)

        return {'loss': loss, 'accuracy': accuracy}
    
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

    def configure_optimizers(self):
        # Only optimize the parameters of your projection layers
        optimizer = torch.optim.Adam(
            [
                {"params": self.vision_proj.parameters()},
                {"params": self.text_proj.parameters()},
            ],
            lr=1e-5,
        )
        return optimizer


