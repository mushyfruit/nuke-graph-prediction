import copy
import time
import logging
import random
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
import torch.amp as amp
import torch.optim as optim
from torch_geometric.loader import DataLoader

from .constants import TrainingStatus, TrainingPhase
from .visualization import LossPlotterThread
from .metrics import compute_top_k_accuracy
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


def setup_dataloaders(dataset: GraphDataset, config: TrainingConfig):
    by_label = {}
    for graph in dataset.examples:
        by_label.setdefault(graph.y.item(), []).append(graph)

    train_dataset, val_dataset = [], []
    rng = random.Random(42)
    for examples in by_label.values():
        if len(examples) < 2:
            continue
        rng.shuffle(examples)
        split = max(1, round(len(examples) * 0.2))
        val_dataset.extend(examples[:split])
        train_dataset.extend(examples[split:])

    if not train_dataset:
        raise ValueError("Training requires a node type with at least two examples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
    )

    return train_loader, val_loader


def train_model_gat(
    dataset: GraphDataset,
    model: NukeGATPredictor,
    memory_fraction: float,
    status_queue: "StatusQueue",
    plot_queue: "StatusQueue",
    config: Optional[TrainingConfig] = None,
):
    if config is None:
        config = TrainingConfig()

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    model = model.to(device)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(memory_fraction)

    log.info("Using {} device".format(device))

    train_loader, val_loader = setup_dataloaders(dataset, config)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        cooldown=2,
        threshold=1e-4,
    )

    scaler = amp.GradScaler(device_type, enabled=device_type == "cuda")

    best_val_accuracy = 0
    best_model_state = None
    start_time = time.time()

    plotter = LossPlotterThread(plot_queue)
    plotter.start()

    try:
        for epoch in range(config.epochs):
            model.train()

            total_loss = 0
            topk_accumulator = {1: 0, 3: 0, 5: 0}
            topk_total = 0

            for batch in train_loader:
                batch = batch.to(device)

                optimizer.zero_grad(set_to_none=True)

                with amp.autocast(device_type=device_type, enabled=device_type == "cuda"):
                    predictions = model(batch)

                    loss = criterion(predictions, batch.y)

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.gradient_clip
                )

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                result = compute_top_k_accuracy(predictions, batch.y)
                for k in result:
                    topk_accumulator[k] += result[k] * batch.y.size(0) / 100.0
                topk_total += batch.y.size(0)

            topk_avg = {k: (v / topk_total) * 100.0 for k, v in topk_accumulator.items()}
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = topk_avg[1]

            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad(), amp.autocast(
                device_type=device_type, enabled=device_type == "cuda"
            ):
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

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = copy.deepcopy(model.state_dict())
                log.info(f"  New best validation accuracy: {best_val_accuracy:.2f}%")

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

            status_queue.safe_put(status_update)
            plot_queue.safe_put(status_update)

        training_time = time.time() - start_time
        log.info(f"\nTraining completed in {training_time:.2f} seconds")
        log.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model
    finally:
        plotter.stop()
        plotter.join()
