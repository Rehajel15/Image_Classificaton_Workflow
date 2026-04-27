from torchinfo import summary
import torch
from torch import nn, save, inference_mode, softmax, argmax
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import os
import random
import time
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# ImageNet mean/std — default un-normalization values for torchvision pretrained models.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible training runs."""
    print(f"----------- Setting seed to {seed} -----------\n")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Create summary function for a model
def model_summary(model:nn.Module, input_size:dict) -> None:
    print("----------- Model summary -----------\n")
    print(f"Gathering information about model...\n")
    # # Get a summary of the model 
    print(summary(model,
                      input_size=input_size, # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
                      verbose=0,
                      col_names=["input_size", "output_size", "num_params", "trainable"],
                      col_width=20,
                      row_settings=["var_names"]
    ))
    print("\n")

# Create function to save state dict of a model
def save_model(model:nn.Module, model_name:str) -> None:
    print("----------- Saving model -----------\n")
    models_path = Path("models")
    # Check if directory for saved models exists
    if not models_path.exists():
        print("Path for saved models not found, creating one...")
        models_path.mkdir(exist_ok=False)

    # Define path for model
    save_path = models_path / f"{model_name}.pth"

    # Save the models state dict
    save(model.state_dict(), save_path)
    print(f"Model '{model_name}' saved in '{save_path}'\n")

# Create a function to create SummaryWriters
def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None) -> SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    print("----------- Creating SummaryWriter -----------\n")

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...\n")
    return SummaryWriter(log_dir=log_dir)

# Plots a confusion matrix
def plot_confusion_matrix(model: nn.Module,
                          test_dataloader: DataLoader,
                          class_names: list,
                          device: str,
                          title: str = "Confusion Matrix"):
    """
    Evaluates the model on the test set and plots a confusion matrix.
    """
    # 1. Set model to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []

    print("----------- Generating Predictions for Confusion Matrix -----------\n")

    # 2. Collect predictions
    with inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            logits = model(X)
            preds = argmax(softmax(logits, dim=1), dim=1)
            
            # Append to lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 3. Create the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 4. Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # 5. Print a detailed classification report (Precision, Recall, F1)
    print("--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))


# Sends a notification to the ntfy chat
def send_notification(topic, message, title="Python Alert"):
    """
    Sends a push notification to a phone via ntfy.sh
    """
    url = f"https://ntfy.sh/{topic}"
    try:
        response = requests.post(
            url,
            data=message.encode('utf-8'),
            headers={
                "Title": title,
                "Priority": "high",
                "Click": f"https://ntfy.sh/{topic}" 
            }
        )
        if response.status_code == 200:
            print(f"Notification sent to topic '{topic}'.")
        else:
            print(f"Failed to send. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while sending: {e}")

# Waits for a response from me in the ntfy app
def wait_for_stop_signal(topic, stop_message: str, continue_message: str,
                         max_retries: int = 5, backoff_seconds: float = 5.0) -> bool:
    """
    Listens to the ntfy topic and returns False for 'stopp' or True for any other command.

    Network errors are retried with exponential backoff. If all retries are exhausted,
    we treat that as a stop (so the model gets saved) but log the failure loudly so the
    user knows training ended due to connectivity, not an explicit stop.
    """
    print(f"Waiting for remote 'Stopp' command on topic: {topic}...")
    url = f"https://ntfy.sh/{topic}/json"

    attempt = 0
    while attempt < max_retries:
        try:
            with requests.get(url, stream=True, timeout=(10, None)) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("event") != "message":
                        continue

                    message_content = data.get("message", "").strip()
                    print(f"Received: '{message_content}'")

                    if message_content.lower() == "stopp":
                        print("Stop signal received. Terminating script...\n")
                        send_notification(topic=topic, message=stop_message, title="Stopping!")
                        return False

                    print("Continuing signal received. Continuing script...\n")
                    send_notification(topic=topic, message=continue_message, title="Continuing!")
                    return True

            # Stream ended without a message — retry.
            print("[ntfy] stream ended without a message, reconnecting...")
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            attempt += 1
            wait = backoff_seconds * (2 ** (attempt - 1))
            print(f"[ntfy] connection error ({e}); retry {attempt}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            print(f"[ntfy] malformed message ignored: {e}")
            continue

    print(f"[ntfy] !!! could not reach ntfy after {max_retries} attempts — defaulting to STOP so the model is saved.\n")
    return False


# Plot n random images from a dataloader
def plot_dataloader_images(dataloader, class_names, n=5, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Plots the first n images from a DataLoader batch.

    mean/std default to ImageNet values; pass the actual normalization values used by
    your transforms if they differ, otherwise the preview will look off.
    """
    images, labels = next(iter(dataloader))

    mean = np.asarray(mean)
    std = np.asarray(std)

    plt.figure(figsize=(15, 5))

    for i in range(n):
        if i >= len(images):
            break

        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(f"Label: {class_names[labels[i]]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()