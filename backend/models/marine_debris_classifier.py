"""
Model 3: Marine Debris Classification
Multi-class classifier for different types of marine debris
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple
from ..training.pipeline import TrainingPipeline


class MarineDebrisDataset(Dataset):
    """Dataset for marine debris classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MarineDebrisClassifier(nn.Module):
    """VGG-based model for marine debris classification"""
    
    def __init__(self, num_classes: int = 6):
        super(MarineDebrisClassifier, self).__init__()
        # Use VGG16 with batch normalization
        self.backbone = models.vgg16_bn(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.features.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class MarineDebrisPipeline(TrainingPipeline):
    """Training pipeline for marine debris classification"""
    
    def build_model(self) -> nn.Module:
        num_classes = self.hyperparameters.get("num_classes", 6)
        return MarineDebrisClassifier(num_classes=num_classes)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create dummy data
        train_images = []
        train_labels = []
        val_images = []
        val_labels = []
        
        train_dataset = MarineDebrisDataset(train_images, train_labels, train_transform)
        val_dataset = MarineDebrisDataset(val_images, val_labels, val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
