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
        hidden_channels,
        heads=8,
        dropout=0.2,
        beta=0.8,
        beta_trainable=True,
    ):
        super().__init__()

        self.gat = GATv2Conv(
            in_channels, out_channels, heads=heads, dropout=dropout, add_self_loops=True
        )

        self.residual_transform = Linear(hidden_channels, out_channels * heads)

        if beta_trainable:
            self.beta = torch.nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))

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
        **kwargs,
    ):
        super().__init__()

        self.heads = heads
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_features = num_features
        self.num_classes = num_classes
        self.beta = beta

        self.feature_encoder = torch.nn.Sequential(
            Linear(num_features, hidden_channels),
            LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

        # GAT layers with multi-head attention, initial res connections.
        self.get_gat_layers()

        # Add layer normalization after each GAT layer
        self.norms = torch.nn.ModuleList(
            [LayerNorm(hidden_channels * heads) for _ in range(num_layers)]
        )

        # Final prediction layers with correct dimensions
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

        self.dropout = dropout
        self.apply(self._init_weights)

    def get_gat_layers(self):
        self.gat_layers = torch.nn.ModuleList()

        self.gat_layers.append(
            InitialResidualGATLayer(
                self.hidden_channels,
                self.hidden_channels,
                self.hidden_channels,
                heads=self.heads,
                dropout=self.dropout,
                beta=self.beta,
            )
        )

        # Middle layers: maintain hidden_channels * heads dimension
        for _ in range(self.num_layers - 1):
            self.gat_layers.append(
                InitialResidualGATLayer(
                    self.hidden_channels * self.heads,
                    self.hidden_channels,
                    self.hidden_channels,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=self.beta,
                )
            )

    def _init_weights(self, module):
        if isinstance(module, Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x_encoded = self.feature_encoder(x)
        x_initial = x_encoded

        x = x_encoded
        for gat, norm in zip(self.gat_layers, self.norms):
            x = gat(x, x_initial, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # [batch_size, hidden_channels * heads]
        x = global_mean_pool(x, data.batch)

        # Final prediction layers
        logits = self.prediction_head(x)

        return logits

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
