"""
Training pipeline orchestration system
"""
import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Base training pipeline for ML models"""
    
    def __init__(
        self,
        model_id: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        data_path: str
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract hyperparameters
        self.batch_size = hyperparameters.get("batch_size", 32)
        self.learning_rate = hyperparameters.get("learning_rate", 0.001)
        self.epochs = hyperparameters.get("epochs", 50)
        self.optimizer_name = hyperparameters.get("optimizer", "adam")
        
    def build_model(self) -> nn.Module:
        """Build the neural network model"""
        raise NotImplementedError("Subclasses must implement build_model")
    
    def prepare_data(self) -> tuple:
        """Prepare training and validation data"""
        raise NotImplementedError("Subclasses must implement prepare_data")
    
    def get_criterion(self) -> nn.Module:
        """Get loss criterion for training. Can be overridden by subclasses."""
        return nn.CrossEntropyLoss()
    
    def is_classification_task(self) -> bool:
        """Check if this is a classification task. Override in regression subclasses."""
        return True
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return {
            "loss": running_loss / len(train_loader),
            "accuracy": 100.0 * correct / total
        }
    
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                
                # Only compute accuracy for classification tasks
                if self.is_classification_task():
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        
        val_accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return {
            "val_loss": running_loss / len(val_loader),
            "val_accuracy": val_accuracy
        }
    
    def train(self, callback=None) -> Dict[str, Any]:
        """Execute the training pipeline"""
        logger.info(f"Starting training for model {self.model_id}")
        
        # Build model
        model = self.build_model().to(self.device)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data()
        
        # Setup training
        # Use CrossEntropyLoss by default, but subclasses can override get_criterion()
        criterion = self.get_criterion()
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_acc = 0.0
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        for epoch in range(self.epochs):
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_metrics = self.validate(model, val_loader, criterion)
            
            # Log metrics
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_accuracy"].append(val_metrics["val_accuracy"])
            
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Loss: {train_metrics['loss']:.4f} - "
                f"Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
            )
            
            # Save best model
            if val_metrics["val_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["val_accuracy"]
                model_path = f"./data/models/{self.model_id}_best.pth"
                torch.save(model.state_dict(), model_path)
            
            # Callback for progress updates
            if callback:
                callback(epoch + 1, self.epochs, train_metrics, val_metrics)
        
        # Save final model
        final_model_path = f"./data/models/{self.model_id}_final.pth"
        torch.save(model.state_dict(), final_model_path)
        
        return {
            "model_path": final_model_path,
            "best_val_accuracy": best_val_acc,
            "history": history,
            "final_metrics": {
                "accuracy": history["val_accuracy"][-1],
                "loss": history["val_loss"][-1]
            }
        }


class TrainingOrchestrator:
    """Orchestrates multiple training jobs"""
    
    def __init__(self, max_concurrent_jobs: int = 5):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs: Dict[str, asyncio.Task] = {}
    
    async def submit_training_job(
        self,
        pipeline: TrainingPipeline,
        callback=None
    ) -> str:
        """Submit a training job"""
        job_id = pipeline.model_id
        
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            raise RuntimeError("Maximum concurrent training jobs reached")
        
        # Create async task
        task = asyncio.create_task(self._run_training(pipeline, callback))
        self.active_jobs[job_id] = task
        
        return job_id
    
    async def _run_training(self, pipeline: TrainingPipeline, callback=None):
        """Run training in background"""
        try:
            result = await asyncio.to_thread(pipeline.train, callback)
            return result
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Remove from active jobs
            if pipeline.model_id in self.active_jobs:
                del self.active_jobs[pipeline.model_id]
    
    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get status of a training job"""
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            if task.done():
                return "completed"
            return "running"
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        if job_id in self.active_jobs:
            self.active_jobs[job_id].cancel()
            del self.active_jobs[job_id]
            return True
        return False
