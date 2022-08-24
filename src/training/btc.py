import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from einops import rearrange

from src.training.transformer import MultiheadSelfAttention, positional_encoding


class BTCSubBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, direction: str, dropout_p: float):
        super().__init__()
        self.msa = MultiheadSelfAttention(dim, n_heads, direction)
        self.cnn = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa(self.norm1(x)) + x
        x = self.dropout(x)
        x = (
            rearrange(
                self.cnn(rearrange(self.norm2(x), "b s c -> b c s")), "b c s -> b s c"
            )
            + x
        )
        return x


class BTCBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout_p: float):
        super().__init__()
        self.left_subblock = BTCSubBlock(dim, n_heads, "left", dropout_p)
        self.right_subblock = BTCSubBlock(dim, n_heads, "right", dropout_p)
        self.projection = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.left_subblock(x)
        right = self.right_subblock(x)
        concatenated = rearrange([left, right], "lr b s c -> b s (lr c)")
        projected = self.projection(concatenated)
        return self.norm(projected)


class BTC(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        dim: int,
        n_heads: int,
        n_blocks: int,
        n_classes: int,
        dropout_p: float,
    ):
        super().__init__()
        # store parameters
        self.input_dim = input_dim
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.dropout_p = dropout_p

        # prepare position encoding
        self.positional_encoding = torch.empty((0, 0))

        # prepare layers
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Linear(input_dim, dim)
        self.blocks = nn.Sequential(
            *[BTCBlock(dim, n_heads, dropout_p) for i in range(n_blocks)]
        )
        self.classification_head = nn.Linear(dim, n_classes)
        self.norm = nn.LayerNorm(dim)

        # prepare metrics
        self.train_accuracy = Accuracy()
        self.validate_accuracy = Accuracy()

    def forward(self, x):
        # generate position encoding if necessary
        if self.positional_encoding.shape[0] < x.shape[1]:
            self.positional_encoding = positional_encoding(x.shape[1], self.dim)

        # dropout input
        x = self.dropout(x)

        # embedd input and add position encoding
        x = self.embedding(x) + self.positional_encoding.to(x.device)

        # transformer blocks
        x = self.norm(self.blocks(x))

        # classification head
        return self.classification_head(x)

    def training_step(self, batch, batch_idx):
        audio, labels = batch
        logits = self(audio)
        loss = torch.nn.functional.cross_entropy(
            rearrange(logits, "b s c -> (b s) c"), rearrange(labels, "b s -> (b s)")
        )

        self.train_accuracy(rearrange(logits, "b s c-> (b s) c"), rearrange(labels, "b s -> (b s)"))
        self.log("loss", loss, on_epoch=True)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        audio, labels = batch
        logits = self(audio)
        self.validate_accuracy(rearrange(logits, "b s c-> (b s) c"), rearrange(labels, "b s -> (b s)"))
        self.log("validate_accuracy", self.validate_accuracy)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


if __name__ == "__main__":
    from torchinfo import summary

    transformer = BTC(144, 128, 4, 8, 25, 0.2)
    print(transformer(torch.rand(5, 100, 144)))
    summary(transformer, input_data=torch.rand(2, 108, 144))
