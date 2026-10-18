import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import LayerNorm, Linear
from torch_geometric.nn import global_mean_pool


class InitialResidualGATLayer(torch.nn.Module):
    """Inspired by Adaptive Depth Graph Attention Network - Zhou et al."""

    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        dropout,
        beta,
    ):
        super().__init__()

        self.gat = GATv2Conv(
            in_channels, out_channels, heads=heads, dropout=dropout, add_self_loops=True
        )

        self.residual_transform = Linear(out_channels, out_channels * heads)

        self.beta = torch.nn.Parameter(torch.tensor(beta))

        self.norm = LayerNorm(out_channels * heads)

    def forward(self, x, x_initial, edge_index):
        gat_out = self.gat(x, edge_index)
        residual = self.residual_transform(x_initial)
        out = gat_out + self.beta * residual
        out = self.norm(out)
        out = F.gelu(out)
        return out


class NukeGATPredictor(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_channels=128,
        num_layers=3,
        heads=8,
        dropout=0.15,
        beta=0.8,
    ):
        super().__init__()

        self.heads = heads
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_features = num_features
        self.num_classes = num_classes

        # Type id column -> 8-dim embedding; the extra zero row is for unknown types.
        self.node_embedding = torch.nn.Embedding(
            num_classes + 1, 8, padding_idx=num_classes
        )
        self.feature_encoder = torch.nn.Sequential(
            Linear(num_features - 1 + 8, hidden_channels),
            LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

        self.gat_layers = torch.nn.ModuleList(
            [
                InitialResidualGATLayer(
                    hidden_channels if i == 0 else hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    beta=beta,
                )
                for i in range(num_layers)
            ]
        )

        final_dim = hidden_channels * heads

        self.prediction_head = torch.nn.Sequential(
            Linear(final_dim, final_dim),
            LayerNorm(final_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            Linear(final_dim, hidden_channels),
            LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, self.num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.1)

    def forward(self, data):
        x = data.x
        type_ids = x[:, 0].long()
        x = torch.cat((self.node_embedding(type_ids), x[:, 1:]), dim=1)
        x_initial = self.feature_encoder(x)
        x = x_initial
        for gat in self.gat_layers:
            x = gat(x, x_initial, data.edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, data.batch)
        return self.prediction_head(x)

    def to_checkpoint(self) -> dict:
        return {
            "num_features": self.num_features,
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "num_heads": self.heads,
            "dropout": self.dropout,
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, num_classes):
        return cls(
            num_features=checkpoint["num_features"],
            num_classes=num_classes,
            hidden_channels=checkpoint["hidden_channels"],
            num_layers=checkpoint["num_layers"],
            heads=checkpoint["num_heads"],
            dropout=checkpoint["dropout"],
        )
