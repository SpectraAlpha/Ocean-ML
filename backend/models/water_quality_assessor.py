"""
Model 4: Water Quality Assessment
Regression model for assessing water quality from images
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple
from ..training.pipeline import TrainingPipeline


class WaterQualityDataset(Dataset):
    """Dataset for water quality assessment"""
    
    def __init__(self, image_paths: List[str], quality_scores: List[float], transform=None):
        self.image_paths = image_paths
        self.quality_scores = quality_scores
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        score = self.quality_scores[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(score, dtype=torch.float32)


class WaterQualityAssessor(nn.Module):
    """DenseNet-based model for water quality assessment"""
    
    def __init__(self, output_dim: int = 5):
        super(WaterQualityAssessor, self).__init__()
        # Use DenseNet121
        self.backbone = models.densenet121(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
        
        # Replace classifier for regression
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # Output quality scores between 0 and 1
        )
    
    def forward(self, x):
        return self.backbone(x)


class WaterQualityPipeline(TrainingPipeline):
    """Training pipeline for water quality assessment"""
    
    def build_model(self) -> nn.Module:
        output_dim = self.hyperparameters.get("output_dim", 5)
        return WaterQualityAssessor(output_dim=output_dim)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
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
        train_scores = []
        val_images = []
        val_scores = []
        
        train_dataset = WaterQualityDataset(train_images, train_scores, train_transform)
        val_dataset = WaterQualityDataset(val_images, val_scores, val_transform)
        
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
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Override train_epoch for regression task"""
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        return {
            "loss": running_loss / len(train_loader),
            "accuracy": 0.0  # Not applicable for regression
        }
    
    def validate(self, model, val_loader, criterion):
        """Override validate for regression task"""
        model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
        
        return {
            "val_loss": running_loss / len(val_loader),
            "val_accuracy": 0.0  # Not applicable for regression
        }
