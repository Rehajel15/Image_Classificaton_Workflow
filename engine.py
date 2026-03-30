from torch import nn, softmax, argmax, inference_mode
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from timeit import default_timer as timer
from torch.utils.tensorboard.writer import SummaryWriter
from helper_functions import send_notification, wait_for_stop_signal

# Create the function to do a train step
def train_step(
        model:nn.Module,
        train_dataloader: DataLoader,
        loss_fn:nn.Module,
        scaler: GradScaler,
        optimizer:Optimizer,
        device:str,
    ) -> tuple[int, int]:

    # Put model into train mode
    model.train()

    # Create values for train_loss and train_acc
    train_loss, train_acc = 0,0

    for batch, (X,y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        # Send data to target device
        X,y = X.to(device), y.to(device)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Activate autocast mode
        with autocast(device_type=device):
            # Send data through model
            y_pred_logits = model(X)

            # Get pred probabilites
            y_pred_probs = softmax(y_pred_logits, dim=1)

            # Get predicition
            y_pred = argmax(y_pred_probs, dim=1)

            # Get loss
            loss = loss_fn(y_pred_logits, y)

            # Add loss to train_loss variable
            train_loss += loss.item()

            # Add accuracy to train_acc variable
            train_acc += ((y_pred == y).sum().item()/len(y_pred))

        # scaler backward propagation
        scaler.scale(loss).backward()

        # Scaler step
        scaler.step(optimizer)

        # Update scaler
        scaler.update()

    # Get the train loss and train_acc for one epoch (training went through every batch)
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)

    # Return train_loss and train_acc
    return train_loss, train_acc


# Create the function to do a train step
def test_step(
        model:nn.Module,
        test_dataloader: DataLoader,
        loss_fn:nn.Module,
        device:str,
    ) -> tuple[int, int]:

    # Send model to eval mode 
    model.eval()

    # Create values for train_loss and train_acc
    test_loss, test_acc = 0,0

    for batch, (X,y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        # Send data to target device
        X,y = X.to(device), y.to(device)

        # Start inference_mode
        with inference_mode():
            # Send data through model
            y_pred_logits = model(X)

            # Get pred probabilites
            y_pred_probs = softmax(y_pred_logits, dim=1)

            # Get predicition
            y_pred = argmax(y_pred_probs, dim=1)

            # Get loss
            loss = loss_fn(y_pred_logits, y)

        # Add loss to train_loss variable
        test_loss += loss.item()

        # Add accuracy to train_acc variable
        test_acc += ((y_pred == y).sum().item()/len(y_pred))

    # Get the train loss and train_acc for one epoch (training went through every batch)
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)

    # Return train_loss and train_acc
    return test_loss, test_acc


# Create the main train function
def train(
        epochs:int,
        model:nn.Module,
        train_dataloader: DataLoader,
        test_dataloader:DataLoader,
        loss_fn:nn.Module,
        scaler: GradScaler,
        optimizer:Optimizer,
        device:str,
        writer: SummaryWriter=None,
        ntfy_topic:str=None,
    ):

    print("----------- Training and testing model -----------\n")
    print(f"Training on device {device}!\n")

    # Create the dict for the whole training results
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    def whole_training_loop():
        for epoch in tqdm(range(1,epochs+1)):
            print(f"-------- Epoch: {epoch} --------\n")

            print(f"-------- Training --------")

            # Save time when the train step begins
            start_train_time = timer()

            # Doing the train step
            train_loss, train_acc = train_step(
                model=model,
                train_dataloader=train_dataloader,
                loss_fn=loss_fn,
                scaler=scaler,
                optimizer=optimizer,
                device=device,
            )
            # Adding the results from the train step to the results dict
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)

            # Save time when the train step ended
            end_train_time = timer()

            # Calculate train time
            train_time = end_train_time - start_train_time

            print(f"-------- Testing --------")

            # Save time when the test step begins
            start_test_time = timer()

            # Doing the test step
            test_loss, test_acc = test_step(
                model=model,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
            )
            # Adding the results from the test step to the results dict
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # Save time when the test step ended
            end_test_time = timer()

            # Calculate test time
            test_time = end_test_time - start_test_time

            # See if there's a writer, if so, log to it
            if writer:
                # Add results to SummaryWriter
                writer.add_scalars(main_tag="Loss",
                                tag_scalar_dict={"train_loss": train_loss,
                                                    "test_loss": test_loss},
                                global_step=epoch)
                writer.add_scalars(main_tag="Accuracy",
                                tag_scalar_dict={"train_acc": train_acc,
                                                    "test_acc": test_acc},
                                global_step=epoch)

            print(f"-------- Epoch {epoch} results: --------\n")
            print(f"Train loss: {train_loss} | Train accuracy: {train_acc}% | Train time: {train_time} | Test loss: {test_loss} | Test accuracy: {test_acc}% | Test time: {test_time}\n")

        print(f"-------- Train and test results after {epochs} epochs: --------\n")
        print(f"{results}\n")

    whole_training_loop()

    # Send notification to ntfy chat
    if ntfy_topic:
        send_notification(topic=ntfy_topic, message=f"Trained model for {epochs} epoch(s)!\n---Results:---\nTrain loss: {results['train_loss'][-1]}\nTrain accuracy: {results['train_acc'][-1]}%\nTest loss: {results['test_loss'][-1]}\nTest accuracy: {results['test_acc'][-1]}% If you would like to train the model for {epochs} more epoch(s), write anything else but 'stopp'.", title="Finished training!")

        continue_training = wait_for_stop_signal(topic=ntfy_topic, stop_message="Stopping the training loop and saving the model...", continue_message=f"Training the model for {epochs} more epochs!")

        while continue_training:
            whole_training_loop()
            send_notification(topic=ntfy_topic, message=f"Trained model for another {epochs} epoch(s)!\n---Results:---\nTrain loss: {results['train_loss'][-1]}\nTrain accuracy: {results['train_acc'][-1]}%\nTest loss: {results['test_loss'][-1]}\nTest accuracy: {results['test_acc'][-1]}% If you would like to train for {epochs} more epoch(s), write anything else but 'stopp'.", title="Finished training!")
            continue_training = wait_for_stop_signal(topic=ntfy_topic, stop_message="Stopping the training loop and saving the model...", continue_message=f"Training the model for {epochs} more epochs!")
    
    # Close the writer (if exists) after entire training loop
    if writer:
        writer.close()

    return results