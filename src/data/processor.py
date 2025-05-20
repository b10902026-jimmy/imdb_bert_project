from datasets import load_dataset
import numpy as np

def get_data_processor(cfg):
    """Load and preprocess data according to config"""
    # Load dataset with original train/test split
    dataset_dict = load_dataset("imdb", keep_in_memory=True)
    train_dataset = dataset_dict["train"]  # Original train split (25,000 samples)
    test_dataset = dataset_dict["test"]    # Original test split (25,000 samples)

    # Apply sampling if specified (for faster testing)
    sample_size = cfg.get("data", {}).get("sample_size", cfg.get("sample_size", 0))
    if sample_size > 0:
        # For test runs, use a very small subset for faster processing
        if sample_size <= 100:  # If it's a small test run
            # Use a fixed seed for reproducibility
            random_seed = cfg.get("data", {}).get("random_seed", cfg.get("random_seed", 42))
            
            # Create balanced train set
            train_pos_indices = [i for i, item in enumerate(train_dataset) if item["label"] == 1][:sample_size // 4]
            train_neg_indices = [i for i, item in enumerate(train_dataset) if item["label"] == 0][:sample_size // 4]
            train_balanced_indices = train_pos_indices + train_neg_indices
            np.random.seed(random_seed)
            np.random.shuffle(train_balanced_indices)
            train_dataset = train_dataset.select(train_balanced_indices)
            
            # Create balanced test set
            test_pos_indices = [i for i, item in enumerate(test_dataset) if item["label"] == 1][:sample_size // 4]
            test_neg_indices = [i for i, item in enumerate(test_dataset) if item["label"] == 0][:sample_size // 4]
            test_balanced_indices = test_pos_indices + test_neg_indices
            np.random.seed(random_seed + 1)  # Different seed for test set
            np.random.shuffle(test_balanced_indices)
            test_dataset = test_dataset.select(test_balanced_indices)
        else:
            # For larger sample sizes, just shuffle and select
            train_size = sample_size // 2
            test_size = sample_size - train_size
            
            train_dataset = train_dataset.shuffle(
                seed=cfg.get("data", {}).get("random_seed", cfg.get("random_seed", 42))
            ).select(range(train_size))
            
            test_dataset = test_dataset.shuffle(
                seed=cfg.get("data", {}).get("random_seed", cfg.get("random_seed", 43))
            ).select(range(test_size))
    
    # Create validation set from train set (10% of train set)
    val_size = cfg.get("data", {}).get("val_size", cfg.get("val_size", 0.1))
    train_val_split = train_dataset.train_test_split(
        test_size=val_size,
        seed=cfg.get("data", {}).get("random_seed", cfg.get("random_seed", 42))
    )
    
    # Return train and validation sets (test set is kept for final evaluation)
    return train_val_split["train"], train_val_split["test"]
