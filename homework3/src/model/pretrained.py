
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import DistilBertModel
from torchvision.models import resnet18


class VisualWSDModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, embedding_dim=768):
        super(VisualWSDModel, self).__init__()
        # Text Embedding
        self.text_embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Image Embedding
        self.image_embedder = resnet18(pretrained=True)
        self.image_embedder.fc = nn.Identity()  # Remove the classification head

        # FC layers to bring text and image embeddings to the same dimension
        self.text_fc = nn.Sequential(
            nn.Linear(768, 512),  # First linear layer (BERT embedding size to common dimension)
            nn.ReLU(),            # Activation function
            nn.Dropout(0.5),      # Dropout for regularization
            nn.Linear(512, embedding_dim)  # Final linear layer to desired embedding_dim
        )
        self.image_fc = nn.Sequential(
            nn.Linear(512, 256),  # First linear layer (ResNet embedding size to common dimension)
            nn.ReLU(),            # Activation function
            nn.Dropout(0.5),      # Dropout for regularization
            nn.Linear(256, embedding_dim)  # Final linear layer to desired embedding_dim
        )

        # Other parameters
        self.learning_rate = learning_rate
        self.criterion = nn.TripletMarginLoss(margin=1.0)

        # Freeze text embedder parameters
        for param in self.text_embedder.parameters():
            param.requires_grad = False

        # Freeze image embedder parameters
        for param in self.image_embedder.parameters():
            param.requires_grad = False


    def forward(self, input_ids, attention_mask, image):
        # Process Texts
        outputs = self.text_embedder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs.last_hidden_state
        # Extract the embedding of the first token ([CLS] token)
        cls_embeddings = text_embeddings[:, 0]

        # Apply the fully connected layer
        text_embeddings = self.text_fc(cls_embeddings)

        # Process Images
        image_embeddings = self.image_embedder(image)
        image_embeddings = self.image_fc(image_embeddings)

        return text_embeddings, image_embeddings

    def training_step(self, batch, batch_idx):
        # Extract data from batch
        input_ids, attention_mask, pos_images, neg_images = batch

        # Forward pass
        pos_text_embeddings, pos_image_embeddings = self(input_ids, attention_mask, pos_images)
        neg_text_embeddings, neg_image_embeddings = self(input_ids, attention_mask, neg_images)

        # Compute triplet loss and log it
        loss = self.criterion(pos_text_embeddings, pos_image_embeddings, neg_image_embeddings)

        # Compute cosine similarities
        pos_similarity = F.cosine_similarity(pos_text_embeddings, pos_image_embeddings)
        neg_similarity = F.cosine_similarity(pos_text_embeddings, neg_image_embeddings)

        # Compute accuracy
        correct_predictions = (pos_similarity > neg_similarity).sum()
        accuracy = correct_predictions.float() / pos_images.size(0)

        # Log loss and accuracy
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Extract data from batch
        input_ids, attention_mask, pos_images, neg_images = batch

        # Forward pass
        pos_text_embeddings, pos_image_embeddings = self(input_ids, attention_mask, pos_images)
        neg_text_embeddings, neg_image_embeddings = self(input_ids, attention_mask, neg_images)

        # Compute triplet loss and log it
        loss = self.criterion(pos_text_embeddings, pos_image_embeddings, neg_image_embeddings)

        # Compute cosine similarities
        pos_similarity = F.cosine_similarity(pos_text_embeddings, pos_image_embeddings)
        neg_similarity = F.cosine_similarity(pos_text_embeddings, neg_image_embeddings)

        # Compute accuracy
        correct_predictions = (pos_similarity > neg_similarity).sum()
        accuracy = correct_predictions.float() / pos_images.size(0)

        # Log loss and accuracy
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Only optimize the fully connected layer parameters
        optimizer = torch.optim.Adam([
                {'params': self.text_fc.parameters()},
                {'params': self.image_fc.parameters()}
            ], lr=self.learning_rate)
        return optimizer