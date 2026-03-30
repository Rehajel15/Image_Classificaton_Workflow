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
    # Get the train and test dir
    train_dir = f"{data_path}/train"
    test_dir = f"{data_path}/test"

    # Create val path, to check for existence
    val_dir_path = Path(f"{data_path}/val")

    # Check if val path exists
    val_dir_exists = val_dir_path.exists()

    # Create transforms
    test_transform = model_weights.transforms()


    if use_data_augmentation:
        train_transform = transforms.Compose([
            # Step 1: Augment the image
            transforms.TrivialAugmentWide(),
            # Step 2: Standard conversion and normalization
            model_weights.transforms()
        ])
    else:
        train_transform = test_transform
    

    # Create train and test dataset
    print("Creating datasets...")
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    
    if val_dir_exists:
        print("Val path detected! Creating val dataset...")
        val_dir = f"{data_path}/val"
        val_dataset = ImageFolder(root=val_dir, transform=test_transform)

    # Get the class names
    class_names = train_dataset.classes

    # Create the dataloader
    print(f"Creating dataloaders with a batch size of {batch_size} image(s) and {num_workers} workers...")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if val_dir_exists:
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Get batch shape
    single_img_shape = train_dataset[0][0].shape
    batch_shape = (train_dataloader.batch_size, *single_img_shape)

    # return everything
    if val_dir_exists:
        print(f"DataLoaders successfully created! Image batch shape: {batch_shape} | Length of train_dataloader: {len(train_dataloader)} batches | Length of test_dataloader: {len(test_dataloader)} batches | Length of val_dataloader: {len(val_dataloader)} batches\n")
        return train_dataloader, test_dataloader, val_dataloader, class_names
    else:
        print(f"DataLoaders successfully created! Image batch shape: {batch_shape} | Length of train_dataloader: {len(train_dataloader)} batches | Length of test_dataloader: {len(test_dataloader)} batches\n")
        return train_dataloader, test_dataloader, class_names
    
    

    
    
    