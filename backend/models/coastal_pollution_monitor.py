"""
Model 5: Coastal Pollution Monitoring
Object detection model for monitoring coastal pollution
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple
from ..training.pipeline import TrainingPipeline


class CoastalPollutionDataset(Dataset):
    """Dataset for coastal pollution monitoring"""
    
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


class CoastalPollutionMonitor(nn.Module):
    """MobileNet-based model for coastal pollution monitoring"""
    
    def __init__(self, num_classes: int = 5):
        super(CoastalPollutionMonitor, self).__init__()
        # Use MobileNetV3 for efficient inference
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-15]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class CoastalPollutionPipeline(TrainingPipeline):
    """Training pipeline for coastal pollution monitoring"""
    
    def build_model(self) -> nn.Module:
        num_classes = self.hyperparameters.get("num_classes", 5)
        return CoastalPollutionMonitor(num_classes=num_classes)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
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
        
        train_dataset = CoastalPollutionDataset(train_images, train_labels, train_transform)
        val_dataset = CoastalPollutionDataset(val_images, val_labels, val_transform)
        
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
