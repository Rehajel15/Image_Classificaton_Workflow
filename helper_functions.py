from torchinfo import summary
from torch import nn, save, inference_mode, softmax, argmax
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import os
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

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
                          device: str):
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
    
    plt.title('Confusion Matrix: Finger Classification')
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
def wait_for_stop_signal(topic, stop_message:str, continue_message:str) -> bool:
    """
    Listens to the ntfy topic and returns False for 'stopp' or True for any other command.
    """
    print(f"Waiting for remote 'Stopp' command on topic: {topic}...")
    url = f"https://ntfy.sh/{topic}/json"
    
    try:
        # stream=True keeps the connection open to wait for new messages
        with requests.get(url, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    
                    # only filter out real messages (events)
                    if data.get("event") == "message":
                        message_content = data.get("message", "").strip()
                        print(f"Received: '{message_content}'")
                        
                        if message_content.lower() == "stopp":
                            print("Stop signal received. Terminating script...\n")
                            send_notification(topic=topic, message=stop_message, title="Stopping!")
                            return False
                        else:
                            print("Continuing signal received. Continuing script...\n")
                            send_notification(topic=topic, message=continue_message, title="Continuing!")
                            return True
    except Exception as e:
        print(f"An error occurred while listening: {e}")

    # default to stop if no valid message was received
    return False


# Plot 5 random image from a dataloader
def plot_dataloader_images(dataloader, class_names, n=5):
    """
    Plots the first n images from a DataLoader batch.
    """
    # 1. Einen Batch holen
    images, labels = next(iter(dataloader))
    
    # 2. Plot erstellen
    plt.figure(figsize=(15, 5))
    
    for i in range(n):
        if i >= len(images): break
        
        # Bild von Tensor (C, H, W) zu NumPy (H, W, C) konvertieren
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Normalisierung rückgängig machen (für ImageNet Werte)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1) # Werte auf [0, 1] begrenzen
        
        # Subplot hinzufügen
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(f"Label: {class_names[labels[i]]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()