import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.amp as amp

import os
import copy
import time
from dataclasses import dataclass

from .constants import MODEL_NAME
from .utilities import save_model_checkpoint, load_model_checkpoint
from .dataset import NukeGraphDataset
from .model import NukeGATPredictor


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 0.05
    gradient_clip: float = 1.5
    label_smoothing: float = 0.1
    num_workers: int = 4

    # Model Parameters
    num_layers: int = 5
    hidden_channels: int = 128
    dropout: float = 0.2


def main(download=False):
    config = TrainingConfig()
    dataset = NukeGraphDataset(PARSED_SCRIPT_DIR, should_download=download)

    print(f"Dataset contains {len(dataset)} graphs")
    print(f"Number of node types: {len(dataset.node_type_to_idx)}")

    model = NukeGATPredictor(
        num_features=4,
        num_classes=len(dataset.node_type_to_idx),
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        num_layers=config.num_layers,
    )
    trained_model = train_model_gat(dataset, model, config)

    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    save_model_checkpoint(trained_model, dataset, save_dir, MODEL_NAME)


def setup_dataloaders(dataset: NukeGraphDataset, config: TrainingConfig):
    train_size = int(0.8 * len(dataset.examples))
    val_size = len(dataset.examples) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset.examples, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        follow_batch=["x"],
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        follow_batch=["x"],
        persistent_workers=True,
    )

    return train_loader, val_loader


def train_model_gat(dataset, model, config=None, memory_fraction=None):
    if config is None:
        config = TrainingConfig()

    # Compile the model prior to training.
    model = torch.compile(model)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    if memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(memory_fraction)

    print("Using {} device".format(device))

    train_loader, val_loader = setup_dataloaders(dataset, config)

    # 1/30/25 - Learned about SGD -> Adam -> Adam w/ Weight Decay
    # Adjusts how we take the step based on gradients
    # Momentum + RMSProp + Weight Decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # 1/31/25 - Learned about CE, fundamental for classification
    # Simplifies to -log(prob) for one-hot, but label smoothing adds constant.
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # 1/31/25 - Learned about basics of schedulers
    # Like meta-controller sitting on top of optimizer, simple.
    # Checks for improvement, if none decreases LR.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        cooldown=2,
        threshold=1e-4,
    )

    scaler = amp.GradScaler(device_type)

    best_val_accuracy = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(config.epochs):
        # Set the model to 'training' mode.
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        # Loop through each batch.
        for batch in train_loader:
            # Move to GPU.
            batch = batch.to(device)

            # Zero out gradients from previous iteration.
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type=device_type):
                # Forward pass through GAT
                predictions = model(batch)

                # Perform the CrossEntropy loss on predictions vs. ground-truth labels.
                loss = criterion(predictions, batch.y)

            # Computes the `.grad` value for each parameter.
            scaler.scale(loss).backward()

            # Unscale before gradient clipping
            scaler.unscale_(optimizer)

            # 1/31/25 - Learned about computed L_2 norm for gradients.
            # If larger than 1.5, then it scales all grads down to maintain
            # the relative proportion of gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.gradient_clip
            )

            # Already has a reference to `model.parameters()`
            # Checks the `.grad` to pass to `AdamW` to calculate.
            scaler.step(optimizer)
            scaler.update()

            # torch.max finds maximum value and index.
            # discard the maximum logit value.
            _, predicted = torch.max(predictions.data, 1)

            # Accumulate the number of samples processed.
            total += batch.y.size(0)

            # Creates boolean tensor, gets value of correct items
            # Unpack the scalar tensor with .item() to get python value.
            correct += (predicted == batch.y).sum().item()

            # Unpack loss scalar tensor to python value.
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad(), amp.autocast(device_type=device_type):
            for batch in val_loader:
                batch = batch.to(device)
                predictions = model(batch)
                loss = criterion(predictions, batch.y)

                val_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                val_total += batch.y.size(0)
                val_correct += (predicted == batch.y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Learning rate adjustment
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  New best validation accuracy: {best_val_accuracy:.2f}%")

        # Print progress with learning rate
        print(f"Epoch {epoch + 1}/{config.epochs}:")
        print(
            f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def load_for_inference(model_dir: str, device="cuda"):
    model_dir = os.path.join(os.path.dirname(__file__), "model/checkpoints")
    model, _, vocab = load_model_checkpoint(model_dir, MODEL_NAME, device)
    model.eval()
    return model, vocab


if __name__ == "__main__":
    main(download=True)
