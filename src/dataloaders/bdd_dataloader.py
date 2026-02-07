"""DataLoader builders for BDD dataset."""
from typing import Optional

from torch.utils.data import DataLoader

from src.datasets.bdd_dataset import BDDDataset
from src.datasets.transforms import collate_fn, get_train_transform, get_val_transform


def make_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    min_bbox_area: int = 16,
    transform=None
) -> DataLoader:
    """
    Create a DataLoader for BDD dataset.
    
    Args:
        root_dir: Root directory containing BDD data
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        min_bbox_area: Minimum bbox area to include
        transform: Optional custom transform
    
    Returns:
        DataLoader instance
    """
    # Use default transforms if none provided
    if transform is None:
        if split == 'train':
            transform = get_train_transform()
        else:
            transform = get_val_transform()
    
    # Create dataset
    dataset = BDDDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        min_bbox_area=min_bbox_area
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


def get_dataloaders(
    root_dir: str,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    num_workers: int = 4,
    min_bbox_area: int = 16
) -> tuple:
    """
    Get train and validation dataloaders.
    
    Returns:
        (train_loader, val_loader)
    """
    train_loader = make_dataloader(
        root_dir=root_dir,
        split='train',
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        min_bbox_area=min_bbox_area
    )
    
    val_loader = make_dataloader(
        root_dir=root_dir,
        split='val',
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        min_bbox_area=min_bbox_area
    )
    
    return train_loader, val_loader
