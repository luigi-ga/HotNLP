import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPTokenizer
from torch.nn.functional import cosine_similarity
import copy


class ClipMocoModel(pl.LightningModule):
    def __init__(self, queue_size=1024, feature_dim=512, embed_dim=128, momentum=0.999):
        """
        Args:
            queue_size: the number of negative samples to store in the queue
            feature_dim: the dimension of the image and text features
            momentum: the momentum factor for updating the queue
        """
        super().__init__()
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.momentum = momentum

        # Initialize tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize CLIP model and projection layers
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_proj = nn.Linear(self.feature_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.feature_dim, self.embed_dim)

        # Freeze the CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Initialize Momentum CLIP model and projection layers
        self.clip_model_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_proj_m = nn.Linear(self.feature_dim, self.embed_dim)
        self.text_proj_m = nn.Linear(self.feature_dim, self.embed_dim)
        
        # Freeze the Momentum CLIP model
        for param in self.clip_model_m.parameters():
            param.requires_grad = False

        # Initialize queues for contrastive learning
        self.register_buffer("image_queue", torch.randn(queue_size, feature_dim))
        self.image_queue = F.normalize(self.image_queue, dim=1)
        self.register_buffer("text_queue", torch.randn(queue_size, feature_dim))
        self.text_queue = F.normalize(self.text_queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def compute_contrastive_loss(self, image_features, text_features, image_features_m, text_features_m):
        # Ensure correct feature dimensions
        assert image_features.dim() == 2, "image_features should be 2-dimensional"
        assert text_features.dim() == 2, "text_features should be 2-dimensional"
        assert image_features_m.dim() == 2, "image_features_m should be 2-dimensional"
        assert text_features_m.dim() == 2, "text_features_m should be 2-dimensional"

        # Compute similarity scores
        sim_i2t = torch.matmul(image_features, text_features_m.T)  # Image to Text
        sim_t2i = torch.matmul(text_features, image_features_m.T)  # Text to Image

        # Labels for the positive pairs
        labels = torch.arange(sim_i2t.size(0), dtype=torch.long, device=sim_i2t.device)

        # InfoNCE loss
        loss_i2t = F.cross_entropy(sim_i2t, labels)
        loss_t2i = F.cross_entropy(sim_t2i, labels)

        # Total loss is the average of the two losses
        loss = (loss_i2t + loss_t2i) / 2
        return loss

    def forward(self, text, images):
        # Tokenize text inputs and send them to the device
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        # Compute text features from CLIP
        text_features = self.clip_model.get_text_features(**text_inputs).unsqueeze(1)
        # Project text features to the embedding space
        text_features = self.text_proj(text_features)
        
        # Process all images and compute image features
        # Assuming images is a list of tensors with shape [batch_size, 10, C, H, W]
        # We want to convert it to [batch_size * 10, C, H, W] to pass through the model
        images = torch.stack(images, dim=1)
        # Reshape images for processing: [batch_size, 10, C, H, W] -> [batch_size * 10, C, H, W]
        images = images.view(-1, *images.size()[2:])
        # Get image features from CLIP
        image_features = self.clip_model.get_image_features(images)
        # Average the image representations
        image_features = image_features.view(-1, 10, self.feature_dim).mean(dim=1)
        # Project image features to the embedding space
        image_features = self.vision_proj(image_features)

        # Reshape image_features back to [batch_size, 10, feature_size]
        batch_size = text_inputs['input_ids'].size(0)
        image_features = image_features.view(batch_size, -1, image_features.size(-1))

        # Repeat the same for the momentum model
        with torch.no_grad():
            self._momentum_update()
            # Compute text features from CLIP
            text_features_m = self.clip_model_m.get_text_features(**text_inputs)
            text_features_m = self.text_proj_m(text_features_m)

            # Get image features from CLIP
            image_features_m = self.clip_model_m.get_image_features(images)
            image_features_m = image_features_m.view(-1, 10, self.feature_dim).mean(dim=1)
            image_features_m = self.vision_proj_m(image_features_m)
            image_features_m = image_features_m.view(batch_size, -1, image_features_m.size(-1))

        # Remove the unnecessary dimension
        image_features = image_features.squeeze(1)  # Shape: [64, 128]
        text_features = text_features.squeeze(1)    # Shape: [64, 128]
        image_features_m = image_features_m.squeeze(1)  # Shape: [64, 128]
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        image_features_m = F.normalize(image_features_m, dim=1)
        text_features_m = F.normalize(text_features_m, dim=1)

        return image_features, text_features, image_features_m, text_features_m

    def training_step(self, batch, batch_idx):
        text, images, gold_label_index = batch
        # Pass inputs to the model
        image_features, text_features, image_features_m, text_features_m = self.forward(text, images)   
        # Calculate contrastive loss and log it
        loss = self.compute_contrastive_loss(image_features, text_features, image_features_m, text_features_m)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return {'loss': loss}
        
    def validation_step(self, batch, batch_idx):
        text, images, gold_label_index = batch
        # Pass inputs to the model
        image_features, text_features, image_features_m, text_features_m = self.forward(text, images)   
        # Calculate contrastive loss and log it
        loss = self.compute_contrastive_loss(image_features, text_features, image_features_m, text_features_m)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        text, images, gold_label_index = batch

        # Get the features from the forward method
        text_features, image_features, _, _ = self.forward(text, images)

        # Calculate cosine similarity scores
        # Note: Since features are already normalized, we can directly calculate the similarity
        scores = torch.matmul(text_features, image_features.T)  # Shape: [batch_size, batch_size]

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
        # Optimize only the projection layer parameters
        params = list(self.vision_proj.parameters()) + list(self.text_proj.parameters()) + \
                list(self.vision_proj_m.parameters()) + list(self.text_proj_m.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)  # You can adjust the learning rate as needed
        return optimizer
    
    def _initialize_momentum_encoder(self):
        # Create a copy of the image encoder
        momentum_encoder = copy.deepcopy(self.clip_model.visual)
        for param in momentum_encoder.parameters():
            param.requires_grad = False
        return momentum_encoder
    
    def _momentum_update(self):
        # Update the momentum encoder's weights
        for param_q, param_k in zip(self.clip_model.parameters(), self.clip_model_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def _dequeue_and_enqueue(self, image_features, text_features):
        # Update the queues with new features
        batch_size = image_features.size(0)
        ptr = int(self.queue_ptr)
        assert self.image_queue.size(0) % batch_size == 0  # for simplicity

        self.image_queue[ptr:ptr + batch_size, :] = image_features
        self.text_queue[ptr:ptr + batch_size, :] = text_features
        ptr = (ptr + batch_size) % self.image_queue.size(0)  # move pointer

        self.queue_ptr[0] = ptr