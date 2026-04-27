from torch import cuda
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from model_builder import create_effnetb0_model
from helper_functions import (
    model_summary,
    save_model,
    plot_confusion_matrix,
    plot_dataloader_images,
    set_seed,
)
from data_setup import create_image_dataloaders
from engine import train

if __name__ == '__main__':
    set_seed(42)

    device = "cuda" if cuda.is_available() else "cpu"

    ntfy_topic = "YOUR_NTFY_TOPIC_HERE"
    epochs = 5
    model_name = "Intel_classification_model"

    model, model_weights = create_effnetb0_model(out_features=6, dropout=0.2, device=device)

    train_dataloader, test_dataloader, val_dataloader, class_names = create_image_dataloaders(
        data_path="data/Intel-Image-Classification",
        use_data_augmentation=True,
        model_weights=model_weights,
        batch_size=16,
        num_workers=2,
    )

    # If no dedicated val/ split exists, fall back to using the test set for per-epoch
    # validation. The final confusion matrix is then on the same data — interpret with care.
    eval_dataloader = val_dataloader if val_dataloader is not None else test_dataloader

    plot_dataloader_images(train_dataloader, class_names, n=5)

    model_summary(model=model, input_size=(1, 3, 224, 224))

    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(device=device)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train(
        epochs=epochs,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=eval_dataloader,
        loss_fn=loss_fn,
        scaler=scaler,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler,
        ntfy_topic=ntfy_topic,
        early_stopping_patience=3,
        checkpoint_dir="models",
        model_name=model_name,
    )

    save_model(model=model, model_name=model_name)

    plot_confusion_matrix(
        model=model,
        test_dataloader=test_dataloader,
        class_names=class_names,
        device=device,
        title="Confusion Matrix: Intel Image Classification",
    )
