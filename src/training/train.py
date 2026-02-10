"""Training pipeline for BDD100K object detection."""
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.detector import BDDDetector, create_model
from src.training.optimizer import create_optimizer
from src.evaluation.evaluate import evaluate_model


class Trainer:
    """Trainer class for object detection model."""
    
    def __init__(
        self,
        model: BDDDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        output_dir: str = 'outputs',
        log_interval: int = 10,
        save_interval: int = 1,
        use_tensorboard: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        """
        Initialize trainer.
        
        Args:
            model: Detection model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to train on
            output_dir: Directory to save checkpoints and logs
            log_interval: Log every N batches
            save_interval: Save checkpoint every N epochs
            use_tensorboard: Whether to use tensorboard logging
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.writer = None
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        loss_components = {
            'loss_classifier': 0.0,
            'loss_box_reg': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Normalize loss by accumulation steps
            losses = losses / self.gradient_accumulation_steps
            
            # Backward pass
            losses.backward()
            
            # Perform optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Accumulate losses (denormalize for logging)
            total_loss += losses.item() * self.gradient_accumulation_steps
            for key, value in loss_dict.items():
                if key in loss_components:
                    loss_components[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses.item(),
                'avg_loss': total_loss / (batch_idx + 1)
            })
            
            # Log to tensorboard
            if self.writer and (batch_idx + 1) % self.log_interval == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('train/loss', losses.item(), global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value.item(), global_step)
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        metrics = {
            'total_loss': avg_loss,
            **loss_components
        }
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass (model returns losses in train mode)
                self.model.train()  # Need train mode to get losses
                loss_dict = self.model(images, targets)
                self.model.eval()
                
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                
                pbar.set_postfix({'loss': losses.item()})
        
        avg_loss = total_loss / num_batches
        
        return {'val_loss': avg_loss}
    
    def train(
        self,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            scheduler: Optional learning rate scheduler
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_one_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch)
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['learning_rate'].append(current_lr)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_metrics['total_loss'], epoch)
                self.writer.add_scalar('epoch/val_loss', val_metrics['val_loss'], epoch)
                self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save latest checkpoint (overwrites previous)
            if epoch % self.save_interval == 0:
                # Remove old latest checkpoint if exists
                latest_path = self.output_dir / 'latest_checkpoint.pth'
                if latest_path.exists():
                    latest_path.unlink()
                self.save_checkpoint('latest_checkpoint.pth')
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best model saved! (val_loss: {self.best_val_loss:.4f})")
            
            print()
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        return self.training_history
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        print(f"Checkpoint loaded from: {checkpoint_path}")
