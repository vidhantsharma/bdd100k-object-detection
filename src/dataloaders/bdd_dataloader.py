"""DataLoader builders for BDD dataset."""
from typing import Optional, Dict

from torch.utils.data import DataLoader, Sampler

from src.datasets.bdd_dataset import BDDDataset
from src.datasets.transforms import collate_fn, get_train_transform, get_val_transform
from src.dataloaders.weighted_sampler import create_weighted_sampler


def make_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    min_bbox_area: int = 16,
    transform=None,
    sampler: Optional[Sampler] = None
) -> DataLoader:
    """
    Create a DataLoader for BDD dataset.
    
    Args:
        root_dir: Root directory containing BDD data
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size
        shuffle: Whether to shuffle data (ignored if sampler is provided)
        num_workers: Number of worker processes
        min_bbox_area: Minimum bbox area to include
        transform: Optional custom transform
        sampler: Optional custom sampler (e.g., WeightedRandomSampler)
    
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
    # Note: shuffle and sampler are mutually exclusive
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),  # Only shuffle if no sampler
        sampler=sampler,
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
    min_bbox_area: int = 16,
    use_weighted_sampling: bool = False,
    class_weights: Optional[Dict[str, float]] = None
) -> tuple:
    """
    Get train and validation dataloaders.
    
    Args:
        root_dir: Root directory containing BDD data
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: Number of worker processes
        min_bbox_area: Minimum bbox area to include
        use_weighted_sampling: Whether to use weighted sampling for training
        class_weights: Dict of class weights (required if use_weighted_sampling=True)
    
    Returns:
        (train_loader, val_loader)
    """
    # Create train sampler if weighted sampling is enabled
    train_sampler = None
    if use_weighted_sampling:
        if class_weights is None:
            raise ValueError(
                "class_weights must be provided when use_weighted_sampling=True.\n"
                "Please run the analysis stage first to compute class weights:\n"
                "  python3 main.py --stage analysis --split train"
            )
        
        # Create temporary dataset to build sampler
        temp_train_dataset = BDDDataset(
            root_dir=root_dir,
            split='train',
            transform=None,  # No transform needed for sampler
            min_bbox_area=min_bbox_area
        )
        train_sampler = create_weighted_sampler(temp_train_dataset, class_weights)
        print(f"✓ Created weighted sampler for training (oversamples rare classes)")
    
    train_loader = make_dataloader(
        root_dir=root_dir,
        split='train',
        batch_size=train_batch_size,
        shuffle=True,  # Will be ignored if sampler is provided
        num_workers=num_workers,
        min_bbox_area=min_bbox_area,
        sampler=train_sampler
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
