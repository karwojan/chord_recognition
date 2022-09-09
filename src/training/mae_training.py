import argparse
import tempfile
import os

import mlflow
import matplotlib.pyplot as plt
import soundfile
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from einops import rearrange, repeat
from torchmetrics import MeanMetric
from torchinfo import summary

from src.training.dataset import SongDataset, SongDatasetConfig
from src.training.model import Transformer
from src.training.evaluate import evaluate


def is_rank_0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def create_argparser():
    parser = argparse.ArgumentParser()
    # dataset
    SongDatasetConfig.add_to_argparser(parser)
    # encoder model
    parser.add_argument("--encoder_dim", type=int, required=True)
    parser.add_argument("--encoder_n_heads", type=int, required=True)
    parser.add_argument("--encoder_n_blocks", type=int, required=True)
    parser.add_argument("--encoder_extra_features_dim", type=int, required=False)
    # decoder model
    parser.add_argument("--decoder_dim", type=int, required=True)
    parser.add_argument("--decoder_n_heads", type=int, required=True)
    parser.add_argument("--decoder_n_blocks", type=int, required=True)
    # training
    parser.add_argument("--masking_ratio", type=float, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=False)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--ddp", action="store_true")
    return parser


def train(args):
    # init torch distributed
    if args.ddp:
        torch.distributed.init_process_group(backend="nccl")

    # init mlflow
    if is_rank_0():
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run(run_name=args.run_name)
        mlflow.log_param("world_size", os.environ.get("WORLD_SIZE", "1"))
        mlflow.log_params(
            {key: value for key, value in args.__dict__.items() if key not in {"experiment_name", "run_name"}}
        )

    # init datasets and data loaders
    train_ds = SongDataset(["train"], SongDatasetConfig.create_from_args(args))
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=5,
        sampler=DistributedSampler(train_ds) if args.ddp else None,
    )

    # prepare models and optimizer
    encoder = Transformer(
        input_dim=train_ds.dim,
        dim=args.encoder_dim,
        n_heads=args.encoder_n_heads,
        n_blocks=args.encoder_n_blocks,
        extra_features_dim=args.encoder_extra_features_dim,
        dropout_p=0.0,
        n_classes=None,
        block_type="transformer",
    ).cuda()
    encoder_embedding = encoder.embedding
    encoder_blocks = encoder.blocks
    decoder = Transformer(
        input_dim=args.encoder_dim,
        dim=args.decoder_dim,
        n_heads=args.decoder_n_heads,
        n_blocks=args.decoder_n_blocks,
        extra_features_dim=None,
        dropout_p=0.0,
        n_classes=train_ds.dim,
        block_type="transformer",
    ).cuda()
    blank = torch.nn.Parameter(torch.rand(size=(args.encoder_dim,)).cuda())
    if args.ddp:
        encoder_embedding = torch.nn.parallel.DistributedDataParallel(encoder_embedding)
        encoder_blocks = torch.nn.parallel.DistributedDataParallel(encoder_blocks)
        decoder = torch.nn.parallel.DistributedDataParallel(decoder)
    optimizer = torch.optim.AdamW(
        list(encoder_embedding.parameters()) + list(encoder_blocks.parameters()) + list(decoder.parameters()) + [blank],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)

    # print model info
    with torch.no_grad():
        data = next(iter(train_dl))[0].cuda()
        summary(encoder_embedding, input_data=data)
        summary(encoder_blocks, input_data=encoder_embedding(data))
        summary(decoder, input_data=encoder_embedding(data))

    # prepare metrics
    loss_metric = MeanMetric().cuda()

    # training loop
    for epoch in tqdm(range(args.n_epochs), unit="epoch"):
        if args.ddp:
            train_dl.sampler.set_epoch(epoch)

        # train
        encoder_embedding.train()
        encoder_blocks.train()
        decoder.train()
        loss_metric.reset()
        for audio, _ in tqdm(train_dl, total=len(train_dl), unit="batch"):
            audio = audio.cuda()
            batch_size, sequence_length = audio.shape[:2]
            masked_sequence_length = int(args.masking_ratio * sequence_length)

            # encoder embedding
            tokens = encoder_embedding(audio)

            # shuffle and cut - mask operation
            shuffle_indices = torch.randperm(sequence_length)
            not_masked_indices = shuffle_indices[:-masked_sequence_length]
            masked_indices = shuffle_indices[-masked_sequence_length:]
            tokens = tokens[:, not_masked_indices]

            # encoder blocks
            tokens = encoder_blocks(tokens)

            # concat and unshuffle
            tokens = torch.cat([tokens, repeat(blank, "c -> b s c", b=batch_size, s=masked_sequence_length)], dim=1)
            tokens = tokens[:, torch.argsort(shuffle_indices)]

            # decoder
            pred_audio = decoder(tokens)

            # MSE loss
            loss = torch.nn.functional.mse_loss(pred_audio[:, masked_indices], audio[:, masked_indices])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_metric(loss)
                if is_rank_0():
                    mlflow.log_metric("train / batch / loss", loss)

        scheduler.step()
        if is_rank_0():
            mlflow.log_metric("train / epoch / loss", loss_metric.compute(), epoch)
            with tempfile.TemporaryDirectory() as tmp_dir:
                # save encoder state dict
                if args.ddp:
                    encoder_embedding_state_dict = encoder_embedding.module.state_dict()
                    encoder_blocks_state_dict = encoder_blocks.module.state_dict()
                else:
                    encoder_embedding_state_dict = encoder_embedding.state_dict()
                    encoder_blocks_state_dict = encoder_blocks.state_dict()
                torch.save(
                    {
                        "encoder_embedding": encoder_embedding_state_dict,
                        "encoder_blocks": encoder_blocks_state_dict,
                    },
                    os.path.join(tmp_dir, "encoder_checkpoint.pt"),
                )

                # save N predictions from last batch
                # (notice that this is random part of dataset in distributed training)
                audio, pred_audio = audio.cpu().detach(), pred_audio.cpu().detach()
                masked_audio = audio.clone().detach()
                masked_audio[:, masked_indices] = 0
                for i in range(3):
                    if args.audio_preprocessing == "cqt":
                        plt.subplot(1, 3, 1)
                        plt.axis("off")
                        plt.imshow(audio[i].T)
                        plt.subplot(1, 3, 2)
                        plt.axis("off")
                        plt.imshow(masked_audio[i].T)
                        plt.subplot(1, 3, 3)
                        plt.axis("off")
                        plt.imshow(pred_audio[i].T)
                        plt.savefig(os.path.join(tmp_dir, f"epoch_{epoch:03}_{i}.png"))
                    elif args.audio_preprocessing == "raw":
                        soundfile.write(
                            os.path.join(tmp_dir, f"epoch_{epoch:03}_{i}_gt.wav"),
                            audio[i, :, :args.hop_size].flatten(), args.sample_rate
                        )
                        soundfile.write(
                            os.path.join(tmp_dir, f"epoch_{epoch:03}_{i}_mgt.wav"),
                            masked_audio[i, :, :args.hop_size].flatten(), args.sample_rate
                        )
                        soundfile.write(
                            os.path.join(tmp_dir, f"epoch_{epoch:03}_{i}_pred.wav"),
                            pred_audio[i, :, :args.hop_size].flatten(), args.sample_rate
                        )
                mlflow.log_artifacts(tmp_dir, "training")


if __name__ == "__main__":
    parser = create_argparser()
    train(parser.parse_args())
