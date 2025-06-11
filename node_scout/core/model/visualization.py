import os
import time
import logging
import threading

import matplotlib.pyplot as plt

from .constants import DirectoryConfig

log = logging.getLogger(__name__)


class LossPlotterThread(threading.Thread):
    def __init__(self, plot_queue, plot_path=None):
        super().__init__()

        log.info("Starting the daemon plotting thread.")
        self.daemon = True
        self.queue = plot_queue

        os.makedirs(DirectoryConfig.TMP_GRAPH_FOLDER, exist_ok=True)
        self.loss_plot_path = DirectoryConfig.LOSS_PLOT
        self.top_k_plot_path = DirectoryConfig.TOPK_PLOT

        self.train_loss = []
        self.val_loss = []
        self.topk_history = {1: [], 3: [], 5: []}

        self._stop_event = threading.Event()

        plt.rcParams.update(
            {
                "axes.facecolor": "#2c2c2c",  # chart panel background
                "figure.facecolor": "#2c2c2c",  # full image background
                "axes.edgecolor": "#cccccc",  # border color
                "axes.labelcolor": "#dddddd",
                "xtick.color": "#aaaaaa",
                "ytick.color": "#aaaaaa",
                "text.color": "#ffffff",
                "legend.frameon": True,
                "legend.facecolor": "#1e1e1e",  # legend background
                "legend.edgecolor": "#ffffff",
                "axes.grid": True,
                "grid.color": "#444444",
            }
        )

    def run(self):
        while not self._stop_event.is_set():
            # Drain any updates from the queue
            while not self.queue.empty():
                status = self.queue.get_nowait()
                log.info(f"Received status {status}")
                if status.training_loss is not None:
                    self.train_loss.append(status.training_loss)
                if status.validation_loss is not None:
                    self.val_loss.append(status.validation_loss)

                if hasattr(status, "topk_accuracy"):
                    for k in self.topk_history:
                        if k in status.topk_accuracy:
                            self.topk_history[k].append(status.topk_accuracy[k])

            # Plot if any data exists
            if self.train_loss:
                log.info("Plotting loss and generating an image.")
                self.plot_losses()

            time.sleep(2)

    def plot_losses(self):
        self._plot_loss_graph()

        if any(len(v) > 0 for v in self.topk_history.values()):
            self._plot_topk_accuracy_history(self.topk_history)

    def _plot_topk_accuracy_history(self, topk_history):
        plt.figure(figsize=(4, 2))

        for k, acc_list in topk_history.items():
            plt.plot(acc_list, label=f"Top-{k}")

        plt.title("Top-k Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.top_k_plot_path)
        plt.close()

    def _plot_loss_graph(self):
        plt.figure(figsize=(4, 2))
        plt.plot(self.train_loss, label="Train Loss")
        if self.val_loss:
            plt.plot(self.val_loss, label="Val Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.loss_plot_path)
        plt.close()

    def stop(self):
        self._stop_event.set()
