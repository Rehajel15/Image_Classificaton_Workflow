from pathlib import Path
from torch import nn, softmax, argmax, inference_mode, save
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from timeit import default_timer as timer
from torch.utils.tensorboard.writer import SummaryWriter
from helper_functions import send_notification, wait_for_stop_signal


def train_step(
        model: nn.Module,
        train_dataloader: DataLoader,
        loss_fn: nn.Module,
        scaler: GradScaler,
        optimizer: Optimizer,
        device: str,
    ) -> tuple[float, float]:

    model.train()
    train_loss, train_acc = 0.0, 0.0

    for batch, (X, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device):
            y_pred_logits = model(X)
            y_pred_probs = softmax(y_pred_logits, dim=1)
            y_pred = argmax(y_pred_probs, dim=1)
            loss = loss_fn(y_pred_logits, y)
            train_loss += loss.item()
            train_acc += ((y_pred == y).sum().item() / len(y_pred))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    return train_loss, train_acc


def eval_step(
        model: nn.Module,
        eval_dataloader: DataLoader,
        loss_fn: nn.Module,
        device: str,
    ) -> tuple[float, float]:
    """One pass over an evaluation dataloader (validation or test)."""

    model.eval()
    eval_loss, eval_acc = 0.0, 0.0

    for batch, (X, y) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        X, y = X.to(device), y.to(device)

        with inference_mode():
            y_pred_logits = model(X)
            y_pred_probs = softmax(y_pred_logits, dim=1)
            y_pred = argmax(y_pred_probs, dim=1)
            loss = loss_fn(y_pred_logits, y)

        eval_loss += loss.item()
        eval_acc += ((y_pred == y).sum().item() / len(y_pred))

    eval_loss /= len(eval_dataloader)
    eval_acc /= len(eval_dataloader)
    return eval_loss, eval_acc


# Backward-compatible alias.
test_step = eval_step


def _save_checkpoint(model: nn.Module, checkpoint_dir: Path, model_name: str, suffix: str) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{model_name}_{suffix}.pth"
    save(model.state_dict(), path)
    return path


def train(
        epochs: int,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_fn: nn.Module,
        scaler: GradScaler,
        optimizer: Optimizer,
        device: str,
        lr_scheduler=None,
        writer: SummaryWriter = None,
        ntfy_topic: str = None,
        early_stopping_patience: int = None,
        checkpoint_dir: str = "models",
        model_name: str = "model",
    ):
    """
    Train + per-epoch validate. Saves the best (lowest val_loss) and the last model
    state_dict to `checkpoint_dir`. If `early_stopping_patience` is set, training
    halts after that many epochs without val_loss improvement.
    """

    print("----------- Training and validating model -----------\n")
    print(f"Training on device {device}!\n")

    if val_dataloader is None:
        raise ValueError("val_dataloader is required. Pass either a real val/ split or your test loader as a substitute.")

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None

    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    state = {
        "best_val_loss": float("inf"),
        "epochs_since_improvement": 0,
        "stopped_early": False,
    }

    def whole_training_loop():
        for epoch in tqdm(range(1, epochs + 1)):
            print(f"-------- Epoch: {epoch} --------\n")

            print("-------- Training --------")
            start_train_time = timer()
            train_loss, train_acc = train_step(
                model=model,
                train_dataloader=train_dataloader,
                loss_fn=loss_fn,
                scaler=scaler,
                optimizer=optimizer,
                device=device,
            )
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            train_time = timer() - start_train_time

            print("-------- Validating --------")
            start_val_time = timer()
            val_loss, val_acc = eval_step(
                model=model,
                eval_dataloader=val_dataloader,
                loss_fn=loss_fn,
                device=device,
            )
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            val_time = timer() - start_val_time

            if lr_scheduler is not None:
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            if writer:
                writer.add_scalars("Loss",
                                   {"train_loss": train_loss, "val_loss": val_loss},
                                   global_step=epoch)
                writer.add_scalars("Accuracy",
                                   {"train_acc": train_acc, "val_acc": val_acc},
                                   global_step=epoch)

            if checkpoint_path is not None:
                _save_checkpoint(model, checkpoint_path, model_name, suffix="last")
                if val_loss < state["best_val_loss"]:
                    state["best_val_loss"] = val_loss
                    state["epochs_since_improvement"] = 0
                    best_path = _save_checkpoint(model, checkpoint_path, model_name, suffix="best")
                    print(f"[checkpoint] new best val_loss={val_loss:.4f} -> saved to {best_path}")
                else:
                    state["epochs_since_improvement"] += 1
            else:
                if val_loss < state["best_val_loss"]:
                    state["best_val_loss"] = val_loss
                    state["epochs_since_improvement"] = 0
                else:
                    state["epochs_since_improvement"] += 1

            print(f"-------- Epoch {epoch} results: --------\n")
            print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Train time: {train_time:.1f}s | "
                  f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Val time: {val_time:.1f}s\n")

            if early_stopping_patience is not None and state["epochs_since_improvement"] >= early_stopping_patience:
                print(f"[early stopping] no val_loss improvement for {early_stopping_patience} epoch(s). Stopping.\n")
                state["stopped_early"] = True
                return

        print(f"-------- Train and val results after {len(results['train_loss'])} epoch(s): --------\n")
        print(f"{results}\n")

    whole_training_loop()

    if ntfy_topic and not state["stopped_early"]:
        send_notification(
            topic=ntfy_topic,
            message=(f"Trained model for {epochs} epoch(s)!\n---Results:---\n"
                     f"Train loss: {results['train_loss'][-1]}\nTrain acc: {results['train_acc'][-1]}\n"
                     f"Val loss: {results['val_loss'][-1]}\nVal acc: {results['val_acc'][-1]}\n"
                     f"Best val loss: {state['best_val_loss']}\n"
                     f"To train another {epochs} epoch(s) reply with anything other than 'stopp'."),
            title="Finished training!",
        )

        continue_training = wait_for_stop_signal(
            topic=ntfy_topic,
            stop_message="Stopping the training loop and saving the model...",
            continue_message=f"Training the model for {epochs} more epochs!",
        )

        while continue_training and not state["stopped_early"]:
            whole_training_loop()
            send_notification(
                topic=ntfy_topic,
                message=(f"Trained model for another {epochs} epoch(s)!\n---Results:---\n"
                         f"Train loss: {results['train_loss'][-1]}\nTrain acc: {results['train_acc'][-1]}\n"
                         f"Val loss: {results['val_loss'][-1]}\nVal acc: {results['val_acc'][-1]}\n"
                         f"Best val loss: {state['best_val_loss']}"),
                title="Finished training!",
            )
            if state["stopped_early"]:
                break
            continue_training = wait_for_stop_signal(
                topic=ntfy_topic,
                stop_message="Stopping the training loop and saving the model...",
                continue_message=f"Training the model for {epochs} more epochs!",
            )

    if writer:
        writer.close()

    return results
