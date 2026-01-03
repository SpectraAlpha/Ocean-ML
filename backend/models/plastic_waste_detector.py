"""
Model 1: Ocean Plastic Waste Detection
CNN-based model for detecting plastic waste in ocean images
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple
from ..training.pipeline import TrainingPipeline


class PlasticWasteDataset(Dataset):
    """Dataset for plastic waste detection"""
    
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


class PlasticWasteDetector(nn.Module):
    """ResNet-based model for plastic waste detection"""
    
    def __init__(self, num_classes: int = 3):
        super(PlasticWasteDetector, self).__init__()
        # Use pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class PlasticWastePipeline(TrainingPipeline):
    """Training pipeline for plastic waste detection"""
    
    def build_model(self) -> nn.Module:
        num_classes = self.hyperparameters.get("num_classes", 3)
        return PlasticWasteDetector(num_classes=num_classes)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders"""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
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
        
        # Create dummy data for demonstration
        # In production, this would load actual data
        train_images = []
        train_labels = []
        val_images = []
        val_labels = []
        
        # Create datasets
        train_dataset = PlasticWasteDataset(train_images, train_labels, train_transform)
        val_dataset = PlasticWasteDataset(val_images, val_labels, val_transform)
        
        # Create data loaders
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
