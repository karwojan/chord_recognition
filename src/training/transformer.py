import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def positional_encoding(sequence_length, dim):
    pos = torch.arange(0, sequence_length)
    wavelengths = 10000 ** (torch.linspace(0, dim, int(dim / 2)) / dim)
    args = rearrange(pos, "s -> s 1") / rearrange(wavelengths, "d -> 1 d")
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, n_heads, dim):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.factor = 1.0 / torch.sqrt(torch.tensor(dim // n_heads))
        assert dim % n_heads == 0

    def forward(self, x):
        # calculate Q, K, V
        Q, K, V = rearrange(
            self.qkv(x),
            "b s (head qkv c) -> qkv (b head) s c",
            qkv=3,
            head=self.n_heads,
        )

        # calculate attention weights
        attention_weights = F.softmax(
            torch.bmm(Q, rearrange(K, "b s c -> b c s")) * self.factor, dim=-1
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
    def __init__(self, n_heads, dim):
        super().__init__()
        self.msa = MultiheadSelfAttention(n_heads, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.msa(x))
        x = self.norm2(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, dim, n_heads, n_blocks, n_classes):
        super().__init__()
        # store parameters
        self.input_dim = input_dim
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_classes = n_classes

        # prepare position encoding
        self.positional_encoding = None

        # prepare layers
        self.embedding = nn.Linear(input_dim, dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_heads, dim) for i in range(n_blocks)]
        )
        self.classification_head = nn.Linear(dim, n_classes)

    def forward(self, x):
        # generate position encoding if necessary
        if (
            self.positional_encoding is None
            or self.positional_encoding.shape[0] < x.shape[1]
        ):
            self.positional_encoding = positional_encoding(x.shape[1], self.dim).to(x.device)

        # embedd input and add position encoding
        x = self.embedding(x) + self.positional_encoding

        # transformer blocks
        x = self.blocks(x)

        # classification head
        return self.classification_head(x)


if __name__ == "__main__":
    from torchinfo import summary

    transformer = Transformer(144, 128, 4, 8, 25)
    summary(transformer, input_data=torch.rand(5, 100, 144))
