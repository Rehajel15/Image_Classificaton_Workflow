from torch import cuda
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.amp import GradScaler
from model_builder import create_effnetb0_model
from helper_functions import model_summary, save_model, plot_confusion_matrix, plot_dataloader_images
from data_setup import create_image_dataloaders
from engine import train

if __name__ == '__main__':
    # Get device
    device = "cuda" if cuda.is_available() else "cpu"

    ntfy_topic = "YOUR_NTFY_TOPIC_HERE"

    # Create the model with transfer learning
    model, model_weights = create_effnetb0_model(out_features=6, dropout=0.2, device=device)

    # Create the dataloaders 
    train_dataloader, test_dataloader, val_dataloader, class_names = create_image_dataloaders(data_path="data/Intel-Image-Classification", use_data_augmentation=True, model_weights=model_weights, batch_size=16, num_workers=2)

    # Plot 5 random images from train_dataloader
    plot_dataloader_images(train_dataloader, class_names, n=5)

    # Get a summary of the model
    model_summary(model=model, input_size=(1,3,224,224))

    # Define loss function, optimizer and GradScaler
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(device=device) # For Mixed Precision

    # Train and test model
    train(
        epochs=5,
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        scaler=scaler,
        optimizer=optimizer,
        device=device,
        ntfy_topic=ntfy_topic
    )

    # Save the model 
    save_model(model=model, model_name="Intel_classification_model")

    # Plot a confusion matrix
    plot_confusion_matrix(
        model=model, 
        test_dataloader=test_dataloader, 
        class_names=class_names, 
        device=device
    )
