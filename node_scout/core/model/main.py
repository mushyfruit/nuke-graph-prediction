import copy
import time
import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from collections import Counter

import torch
import torch.amp as amp
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from .constants import MODEL_NAME, DirectoryConfig, TrainingStatus, TrainingPhase
from .visualization import LossPlotterThread
from .metrics import compute_top_k_accuracy
from .utilities import save_model_checkpoint
from .dataset.dataset import GraphDataset
from .gnn.gat import NukeGATPredictor

if TYPE_CHECKING:
    from .queue import StatusQueue

log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    gradient_clip: float = 1.5
    label_smoothing: float = 0.05
    num_workers: int = 4

    # Model Parameters
    num_heads: int = 8
    num_layers: int = 4
    hidden_channels: int = 128
    dropout: float = 0.3


def main():
    config = TrainingConfig()

    dataset = GraphDataset(DirectoryConfig.MODEL_DATA_FOLDER, force_rebuild=True)

    log.info(f"Dataset contains {len(dataset)} graphs")
    log.info(f"Number of node types: {len(dataset.vocab)}")

    model = NukeGATPredictor(
        num_features=dataset.num_features,
        num_classes=len(dataset.vocab),
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        heads=config.num_heads,
        num_layers=config.num_layers,
    )
    trained_model = train_model_gat(dataset, model, config)
    save_model_checkpoint(trained_model, MODEL_NAME)


def setup_dataloaders(dataset: GraphDataset, config: TrainingConfig):
    label_counts = Counter([g.y.item() for g in dataset.examples])
    valid_labels = {label for label, count in label_counts.items() if count >= 2}

    filtered_examples = [g for g in dataset.examples if g.y.item() in valid_labels]
    labels = [g.y.item() for g in filtered_examples]

    # Ensure an equal representation of node classes in test/validation splits.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(filtered_examples, labels))

    train_dataset = [dataset.examples[i] for i in train_idx]
    val_dataset = [dataset.examples[i] for i in val_idx]

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


def train_model_gat(
    dataset: GraphDataset,
    model: NukeGATPredictor,
    config: Optional[TrainingConfig] = None,
    memory_fraction: Optional[float] = None,
    status_queue: Optional["StatusQueue"] = None,
    plot_queue: Optional["StatusQueue"] = None,
):
    if config is None:
        config = TrainingConfig()

    # Compile the model prior to training.
    if status_queue:
        status_queue.safe_put(
            TrainingStatus(TrainingPhase.TRAINING, label="Compiling model...")
        )

    plotter = None
    if plot_queue:
        plotter = LossPlotterThread(plot_queue=plot_queue)
        plotter.start()

    model = torch.compile(model)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    model = model.to(device)

    torch._dynamo.config.capture_scalar_outputs = True
    torch.backends.cudnn.benchmark = True

    if memory_fraction is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(memory_fraction)

    log.info("Using {} device".format(device))

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
        topk_accumulator = {1: 0, 3: 0, 5: 0}
        topk_total = 0

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

            result = compute_top_k_accuracy(predictions, batch.y)
            for k in result:
                topk_accumulator[k] += result[k] * batch.y.size(0) / 100.0
            topk_total += batch.y.size(0)

            del batch, predictions, loss, predicted

        topk_avg = {k: (v / topk_total) * 100.0 for k, v in topk_accumulator.items()}
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
            log.info(f"  New best validation accuracy: {best_val_accuracy:.2f}%")

        # Print progress with learning rate
        log.info(f"Epoch {epoch + 1}/{config.epochs}:")
        log.info(
            f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )
        log.info(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        log.info(f"  Learning Rate: {current_lr:.2e}")

        current_epoch = epoch + 1
        status_update = TrainingStatus(
            TrainingPhase.TRAINING,
            current_epoch=current_epoch,
            total_epochs=config.epochs,
            training_loss=avg_train_loss,
            training_accuracy=train_accuracy,
            validation_loss=avg_val_loss,
            validation_accuracy=val_accuracy,
            topk_accuracy=topk_avg,
            progress=float(current_epoch / config.epochs),
            label=f"Training Model: {current_epoch}/{config.epochs}",
        )

        if status_queue:
            status_queue.safe_put(status_update)

        if plot_queue:
            plot_queue.safe_put(status_update)

    training_time = time.time() - start_time
    log.info(f"\nTraining completed in {training_time:.2f} seconds")
    log.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")

    if plotter:
        plotter.stop()
        plotter.join()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


if __name__ == "__main__":
    main()
