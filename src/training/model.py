import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from einops import rearrange, repeat


def positional_encoding(sequence_length: int, dim: int) -> torch.Tensor:
    pos = torch.arange(0, sequence_length)
    wavelengths = 10000 ** (torch.linspace(0, dim, int(dim / 2)) / dim)
    args = rearrange(pos, "s -> s 1") / rearrange(wavelengths, "d -> 1 d")
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, mask: str = None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.mask = mask
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.factor = 1.0 / torch.sqrt(torch.tensor(dim // n_heads))
        assert dim % n_heads == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # calculate Q, K, V
        Q, K, V = rearrange(
            self.qkv(x),
            "b s (head qkv c) -> qkv (b head) s c",
            qkv=3,
            head=self.n_heads,
        )

        # optional masking
        b, s, _ = Q.shape
        mask = torch.zeros((b, s, s)).to(x.device)
        if self.mask == "right":
            mask[
                :,
                repeat(torch.arange(s), "s -> S s", S=s)
                > repeat(torch.arange(s), "s -> s S", S=s),
            ] = -torch.inf
        elif self.mask == "left":
            mask[
                :,
                repeat(torch.arange(s), "s -> S s", S=s)
                < repeat(torch.arange(s), "s -> s S", S=s),
            ] = -torch.inf

        # calculate attention weights
        attention_weights = F.softmax(
            torch.bmm(Q, rearrange(K, "b s c -> b c s")) * self.factor + mask, dim=-1
        )

        # apply attention
        output = self.out(
            rearrange(
                torch.bmm(attention_weights, V),
                "(b head) s c -> b s (head c)",
                head=self.n_heads,
            )
        )
        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout_p: float):
        super().__init__()
        self.msa = MultiheadSelfAttention(dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa(self.norm1(x)) + x
        x = self.dropout1(x)
        x = self.mlp(self.norm2(x)) + x
        x = self.dropout2(x)
        return x


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
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa(self.norm1(x)) + x
        x = self.dropout1(x)
        x = (
            rearrange(
                self.cnn(rearrange(self.norm2(x), "b s c -> b c s")), "b c s -> b s c"
            )
            + x
        )
        x = self.dropout2(x)
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


class Transformer(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        dim: int,
        n_heads: int,
        n_blocks: int,
        n_classes: int,
        block_type: str,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        # store parameters
        self.input_dim = input_dim
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.block_type = block_type
        self.dropout_p = dropout_p

        # prepare position encoding
        self.positional_encoding = torch.empty((0, 0))

        # prepare layers
        if block_type == "transformer":
            block = TransformerBlock
        elif block_type == "btc":
            block = BTCBlock
        else:
            raise ValueError(block_type)
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Linear(input_dim, dim)
        self.blocks = nn.Sequential(
            *[block(dim, n_heads, dropout_p) for i in range(n_blocks)]
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
        self.log("loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        audio, labels = batch
        logits = self(audio)
        self.validate_accuracy(rearrange(logits, "b s c-> (b s) c"), rearrange(labels, "b s -> (b s)"))
        self.log("validate_accuracy", self.validate_accuracy, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


if __name__ == "__main__":
    from torchinfo import summary

    transformer = Transformer(144, 128, 4, 8, 25, "btc", 0.2)
    print(transformer(torch.rand(5, 100, 144)))
    summary(transformer, input_data=torch.rand(2, 108, 144))
