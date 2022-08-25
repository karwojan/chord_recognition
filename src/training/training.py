import argparse

import os
import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from torchmetrics import Accuracy, MeanMetric

from src.training.dataset import SongDataset
from src.training.preprocessing import CQTPreprocessing
from src.training.model import Transformer
from src.training.evaluate import evaluate


def is_rank_0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def log_metric(key: str, value: float, step: int = None):
    if is_rank_0():
        mlflow.log_metric(key, value, step)


def create_argparser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--frame_size", type=int, required=True)
    parser.add_argument("--hop_size", type=int, required=True)
    parser.add_argument("--frames_per_item", type=int, required=True)
    parser.add_argument("--pitch_shift_augment", action="store_true")

    # model
    parser.add_argument("--model_dim", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--n_blocks", type=int, required=True)
    parser.add_argument(
        "--block_type", type=str, choices=["btc", "transformer"], required=True
    )
    parser.add_argument("--dropout_p", type=float, required=True)

    # training
    parser.add_argument("--experiment_name", type=str, required=True)
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
        mlflow.log_param("world_size", os.environ.get("WORLD_SIZE", "1"))
        mlflow.log_params(args.__dict__)

    # init datasets and data loaders
    ds_kwargs = {
        "sample_rate": args.sample_rate,
        "frame_size": args.frame_size,
        "hop_size": args.hop_size,
        "audio_preprocessing": CQTPreprocessing(),
        "labels_vocabulary": "maj_min",
        "subsets": ["isophonics", "robbie_williams", "uspop"],
    }
    train_ds = SongDataset(
        ["train"],
        **ds_kwargs,
        frames_per_item=args.frames_per_item,
        pitch_shift_augment=args.pitch_shift_augment
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=5
    )
    validate_ds = SongDataset(
        ["validate"], **ds_kwargs, frames_per_item=args.frames_per_item
    )
    validate_dl = DataLoader(
        validate_ds, batch_size=args.batch_size, shuffle=False, num_workers=5
    )

    # prepare model and optimizer
    model = Transformer(
        train_ds.dim,
        args.model_dim,
        args.n_heads,
        args.n_blocks,
        train_ds.n_classes,
        block_type=args.block_type,
        dropout_p=args.dropout_p,
    ).cuda()
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)

    # prepare metrics
    train_accuracy = Accuracy().cuda()
    validate_accuracy = Accuracy().cuda()
    loss_metric = MeanMetric().cuda()

    # training loop
    for epoch in tqdm(range(args.n_epochs), unit="epoch"):
        # train
        model.train()
        loss_metric.reset()
        train_accuracy.reset()
        for audio, labels in tqdm(train_dl, total=len(train_dl), unit="batch"):
            audio, labels = audio.cuda(), labels.cuda()
            logits = model(audio)
            loss = torch.nn.functional.cross_entropy(
                rearrange(logits, "b s c -> (b s) c"), rearrange(labels, "b s -> (b s)")
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_metric(loss)
                train_accuracy(rearrange(logits, "b s c-> (b s) c"), rearrange(labels, "b s -> (b s)"))

        scheduler.step()
        log_metric("train / epoch / loss", loss_metric.compute(), epoch)
        log_metric("train / epoch / accuracy", train_accuracy.compute(), epoch)

        # validate
        model.eval()
        validate_accuracy.reset()
        for audio, labels in tqdm(validate_dl, total=len(validate_dl), unit="batch"):
            with torch.no_grad():
                audio, labels = audio.cuda(), labels.cuda()
                validate_accuracy(rearrange(model(audio), "b s c-> (b s) c"), rearrange(labels, "b s -> (b s)"))
        log_metric("validate / epoch / accuracy", validate_accuracy.compute(), epoch)

    # evaluate model
    if is_rank_0():
        model.eval()
        evaluate(
            SongDataset(["train"], **ds_kwargs, frames_per_item=0),
            model,
            "train_ds_evaluation",
            args.frames_per_item,
        )
        evaluate(
            SongDataset(["validate"], **ds_kwargs, frames_per_item=0),
            model,
            "validate_ds_evaluation",
            args.frames_per_item,
        )


if __name__ == "__main__":
    parser = create_argparser()
    train(parser.parse_args())
