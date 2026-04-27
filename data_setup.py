from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
from os import cpu_count
from os import listdir
import shutil


def organize_dataset(base_path):
    """
    Organizes the Koryakinp/Fingers dataset into subfolders based on the labels 
    in the filenames (e.g., '0L' -> folder '0').
    """
    print("----------- Organize fingers dataset -----------\n")

    # Define the sets to organize
    for split in ["train", "test"]:
        split_path = Path(base_path) / split
        if not split_path.exists():
            print(f"Path not found: {split_path}")
            continue

        print(f"Organizing {split} set...")

        # Get all image files in the current split folder
        images = [f for f in listdir(split_path) if f.endswith('.png')]

        for img_name in images:
            # The label is the character before 'L.png' or 'R.png'
            # Example: ..._3L.png -> the finger count is '3'
            finger_count = img_name.split('_')[-1][0] 

            # Create class directory if it doesn't exist
            class_dir = split_path / finger_count
            class_dir.mkdir(exist_ok=True)

            # Move the file into the class directory
            src = split_path / img_name
            dst = class_dir / img_name
            shutil.move(src, dst)

    print("Success! Dataset is now ready for ImageFolder\n")


def create_image_dataloaders(
        data_path:str,
        use_data_augmentation:bool=True,
        model_weights=None,
        batch_size:int = 1,
        num_workers:int = cpu_count()):

    print("----------- Creating dataloaders -----------\n")
    print("Searching for train, test and val directory...")
    if model_weights is None:
        raise ValueError("model_weights is required so the dataloader transforms match the pretrained model.")

    train_dir = f"{data_path}/train"
    test_dir = f"{data_path}/test"
    val_dir_path = Path(f"{data_path}/val")
    val_dir_exists = val_dir_path.exists()

    eval_transform = model_weights.transforms()

    if use_data_augmentation:
        train_transform = transforms.Compose([
            transforms.TrivialAugmentWide(),
            eval_transform,
        ])
    else:
        train_transform = eval_transform

    print("Creating datasets...")
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=eval_transform)
    val_dataset = ImageFolder(root=str(val_dir_path), transform=eval_transform) if val_dir_exists else None

    class_names = train_dataset.classes

    print(f"Creating dataloaders with a batch size of {batch_size} image(s) and {num_workers} workers...")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_dataset is not None else None

    single_img_shape = train_dataset[0][0].shape
    batch_shape = (train_dataloader.batch_size, *single_img_shape)

    val_msg = f" | Length of val_dataloader: {len(val_dataloader)} batches" if val_dataloader is not None else " | No val_dataloader (no val/ directory found)"
    print(f"DataLoaders successfully created! Image batch shape: {batch_shape} | Length of train_dataloader: {len(train_dataloader)} batches | Length of test_dataloader: {len(test_dataloader)} batches{val_msg}\n")

    return train_dataloader, test_dataloader, val_dataloader, class_names
    
    

    
    
    