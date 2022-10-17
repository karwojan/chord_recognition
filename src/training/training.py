import argparse
from dataclasses import replace

import os
import mlflow
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from einops import rearrange
from torchmetrics import Accuracy, MeanMetric
from torchinfo import summary

from src.training.dataset import SongDataset, SongDatasetConfig, song_dataset_collate_fn
from src.training.model import Transformer
from src.training.evaluate import evaluate


def is_rank_0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def log_metric(key: str, value: float, step: int = None):
    if is_rank_0():
        mlflow.log_metric(key, value, step)


def load_pretrained_encoder_weights(args, encoder_embedding, encoder_blocks):
    if args.pretrained_encoder_run_name is not None:
        run_id = (
            mlflow.search_runs(
                experiment_names=[args.experiment_name],
                filter_string=f'tags.mlflow.runName = "{args.pretrained_encoder_run_name}"',
            )
            .iloc[0]
            .run_id
        )
        args.pretrained_encoder_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="training/encoder_checkpoint.pt"
        )
    if args.pretrained_encoder_path is not None:
        print(f"Loading pretrained encoder from {args.pretrained_encoder_path}")
        encoder_state_dict = torch.load(args.pretrained_encoder_path)
        encoder_embedding.load_state_dict(encoder_state_dict["encoder_embedding"])
        encoder_blocks.load_state_dict(encoder_state_dict["encoder_blocks"])


def worker_init_fn(worker_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % np.iinfo(np.int32).max)


def create_argparser():
    parser = argparse.ArgumentParser()

    # dataset
    SongDatasetConfig.add_to_argparser(parser)

    # model
    parser.add_argument("--model_dim", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--n_blocks", type=int, required=True)
    parser.add_argument("--block_type", type=str, choices=["btc", "transformer"], required=True)
    parser.add_argument("--dropout_p", type=float, required=True)
    parser.add_argument("--extra_features_dim", type=int, required=False)
    parser.add_argument("--pretrained_encoder_path", type=str, required=False)
    parser.add_argument("--pretrained_encoder_run_name", type=str, required=False)

    # training
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
    song_dataset_config = SongDatasetConfig.create_from_args(args)
    train_ds = SongDataset(["train"], song_dataset_config)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=5,
        sampler=DistributedSampler(train_ds) if args.ddp else None,
        collate_fn=song_dataset_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    validate_ds = SongDataset(["validate"], replace(song_dataset_config, pitch_shift_augment=False, song_multiplier=1))
    validate_dl = DataLoader(
        validate_ds,
        batch_size=args.batch_size,
        num_workers=5,
        sampler=DistributedSampler(validate_ds) if args.ddp else None,
        collate_fn=song_dataset_collate_fn,
        worker_init_fn=worker_init_fn
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
        extra_features_dim=args.extra_features_dim,
    ).cuda()
    load_pretrained_encoder_weights(args, model.embedding, model.blocks)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)

    # print model info
    summary(model, input_data=next(iter(train_dl))[0])

    # prepare metrics
    train_accuracy = Accuracy().cuda()
    validate_accuracy = Accuracy().cuda()
    loss_metric = MeanMetric().cuda()

    # training loop
    for epoch in tqdm(range(args.n_epochs), unit="epoch"):
        if args.ddp:
            train_dl.sampler.set_epoch(epoch)
            validate_dl.sampler.set_epoch(epoch)

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
                train_accuracy(
                    rearrange(logits, "b s c-> (b s) c"),
                    rearrange(labels, "b s -> (b s)"),
                )

        scheduler.step()
        log_metric("train / epoch / loss", loss_metric.compute(), epoch)
        log_metric("train / epoch / accuracy", train_accuracy.compute(), epoch)

        # validate
        model.eval()
        validate_accuracy.reset()
        for audio, labels in tqdm(validate_dl, total=len(validate_dl), unit="batch"):
            with torch.no_grad():
                audio, labels = audio.cuda(), labels.cuda()
                validate_accuracy(
                    rearrange(model(audio), "b s c-> (b s) c"),
                    rearrange(labels, "b s -> (b s)"),
                )
        log_metric("validate / epoch / accuracy", validate_accuracy.compute(), epoch)

    # evaluate model
    if is_rank_0():
        model.eval()
        song_dataset_config_eval = replace(
            song_dataset_config, frames_per_item=0, pitch_shift_augment=False, song_multiplier=1
        )
        evaluate(
            SongDataset(["train"], song_dataset_config_eval),
            model,
            "train_ds_evaluation",
            args.frames_per_item,
        )
        evaluate(
            SongDataset(["validate"], song_dataset_config_eval),
            model,
            "validate_ds_evaluation",
            args.frames_per_item,
        )


if __name__ == "__main__":
    parser = create_argparser()
    train(parser.parse_args())
