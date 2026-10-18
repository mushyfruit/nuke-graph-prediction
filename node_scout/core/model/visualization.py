import os
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import DirectoryConfig


class LossPlotterThread(threading.Thread):
    def __init__(self, plot_queue):
        super().__init__()

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
                "axes.facecolor": "#2c2c2c",
                "figure.facecolor": "#2c2c2c",
                "axes.edgecolor": "#cccccc",
                "axes.labelcolor": "#dddddd",
                "xtick.color": "#aaaaaa",
                "ytick.color": "#aaaaaa",
                "text.color": "#ffffff",
                "legend.frameon": True,
                "legend.facecolor": "#1e1e1e",
                "legend.edgecolor": "#ffffff",
                "axes.grid": True,
                "grid.color": "#444444",
            }
        )

    def run(self):
        while not self._stop_event.is_set() or not self.queue.empty():
            updated = False
            while not self.queue.empty():
                status = self.queue.get_nowait()
                updated = True
                self.train_loss.append(status.training_loss)
                self.val_loss.append(status.validation_loss)
                for k in self.topk_history:
                    self.topk_history[k].append(status.topk_accuracy[k])

            if updated:
                self._plot_loss_graph()
                self._plot_topk_accuracy_history()

            self._stop_event.wait(0.1)

    def _plot_topk_accuracy_history(self):
        plt.figure(figsize=(4, 2))

        for k, acc_list in self.topk_history.items():
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
        plt.plot(self.val_loss, label="Val Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.loss_plot_path)
        plt.close()

    def stop(self):
        self._stop_event.set()
