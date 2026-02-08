"""Optimizer and learning rate scheduler utilities."""
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Optional


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = 'sgd',
    learning_rate: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('sgd', 'adam', 'adamw')
        learning_rate: Initial learning rate
        momentum: Momentum factor (for SGD)
        weight_decay: Weight decay (L2 penalty)
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from: sgd, adam, adamw")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'step',
    step_size: int = 3,
    gamma: float = 0.1,
    milestones: Optional[list] = None,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler ('step', 'multistep', 'cosine', 'plateau', 'none')
        step_size: Period of learning rate decay (for StepLR)
        gamma: Multiplicative factor of learning rate decay
        milestones: List of epoch indices for MultiStepLR
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Scheduler instance or None
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'none' or scheduler_name is None:
        return None
    elif scheduler_name == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            **kwargs
        )
    elif scheduler_name == 'multistep':
        if milestones is None:
            milestones = [8, 11]
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
            **kwargs
        )
    elif scheduler_name == 'cosine':
        T_max = kwargs.pop('T_max', 50)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            **kwargs
        )
    elif scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=gamma,
            patience=kwargs.pop('patience', 3),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler

