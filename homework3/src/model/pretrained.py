
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import DistilBertModel, DistilBertTokenizer
from torchvision.models import resnet18
from torch.nn.functional import cosine_similarity


class VisualWSDModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, projection_dim=128):
        super(VisualWSDModel, self).__init__()
        self.projection_dim = projection_dim
        # Text Embedding
        self.text_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # Image Embedding
        self.image_embedder = resnet18(pretrained=True)
        self.image_embedder.fc = nn.Identity()  # Remove the classification head

        # Projection layers
        self.text_proj = nn.Linear(768, projection_dim)  # Project text embeddings
        self.image_proj = nn.Linear(512, projection_dim)  # Project image embeddings

        # Other parameters
        self.learning_rate = learning_rate

        # Freeze parameters of embedders
        for param in self.text_embedder.parameters():
            param.requires_grad = False
        for param in self.image_embedder.parameters():
            param.requires_grad = False

    def forward(self, text, images):
        # Process Texts
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids, attention_mask = inputs.input_ids.to(self.device), inputs.attention_mask.to(self.device)
        text_outputs = self.text_embedder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token
        text_embeddings = self.text_proj(text_embeddings)  # Project text embeddings

        # Process Images
        images = torch.stack(images).view(-1, *images[0].size()[1:])  # Flatten list of image batches for processing
        image_embeddings = self.image_embedder(images)
        image_embeddings = self.image_proj(image_embeddings)  # Project image embeddings
        image_embeddings = image_embeddings.view(-1, 10, self.projection_dim)  # Reshape back to (batch_size, num_images, projection_dim)

        return text_embeddings, image_embeddings

    def compute_margin_ranking_loss(self, scores, gold_label_index):
        # Similar logic to ClipModelProj for computing the margin ranking loss
        positive_scores = scores.gather(1, gold_label_index.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(1, gold_label_index.unsqueeze(1), False)
        negative_scores = scores[mask].view(scores.size(0), -1)
        positive_scores = positive_scores.unsqueeze(1).expand_as(negative_scores)
        target = torch.ones_like(positive_scores)
        loss_fn = nn.MarginRankingLoss(margin=0.1)
        loss = loss_fn(positive_scores, negative_scores, target)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        labels, images, gold_label_index = batch
        text_embeddings, image_embeddings = self.forward(labels, images)
        scores = cosine_similarity(text_embeddings.unsqueeze(1), image_embeddings, dim=2)
        loss = self.compute_margin_ranking_loss(scores, gold_label_index)
        
        # Accuracy is the number of correct predictions divided by the number of predictions
        accuracy = (scores.argmax(dim=1) == gold_label_index).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        
        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, batch, batch_idx):
        labels, images, gold_label_index = batch
        text_embeddings, image_embeddings = self.forward(labels, images)
        scores = cosine_similarity(text_embeddings.unsqueeze(1), image_embeddings, dim=2)
        loss = self.compute_margin_ranking_loss(scores, gold_label_index)
        
        # Accuracy for validation
        accuracy = (scores.argmax(dim=1) == gold_label_index).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        
        return {'loss': loss, 'accuracy': accuracy}

    def test_step(self, batch, batch_idx):
        labels, images, gold_label_index = batch
        text_embeddings, image_embeddings = self.forward(labels, images)
        scores = cosine_similarity(text_embeddings.unsqueeze(1), image_embeddings, dim=2)

        # Calculate H@1
        top_pred = scores.argmax(dim=1)
        h_at_1 = (top_pred == gold_label_index).float().mean()

        # Calculate MRR
        rank = (scores.argsort(descending=True).argsort(dim=1) == gold_label_index.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        mrr = (1. / rank.float()).mean()

        # Log metrics for testing
        self.log('test_h_at_1', h_at_1, on_epoch=True, prog_bar=True)
        self.log('test_mrr', mrr, on_epoch=True, prog_bar=True)

        return {'h_at_1': h_at_1, 'mrr': mrr}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.text_proj.parameters()},
            {'params': self.image_proj.parameters()}
        ], lr=self.learning_rate)
        return optimizer
    

