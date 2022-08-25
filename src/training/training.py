import argparse
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
import pytorch_lightning as pl

from src.training.dataset import SongDataset
from src.training.preprocessing import CQTPreprocessing
from src.training.model import Transformer
from src.training.evaluate import evaluate


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
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    return parser


def train(args):
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

    # prepare model
    model = Transformer(
        train_ds.dim,
        args.model_dim,
        args.n_heads,
        args.n_blocks,
        train_ds.n_classes,
        block_type=args.block_type,
        dropout_p=args.dropout_p,
    )

    # train model
    logger = pl.loggers.MLFlowLogger(experiment_name="btc_custom_implementation")
    logger.log_hyperparams(args)
    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator="gpu",
        strategy="ddp",
        devices=1,
        num_nodes=int(os.environ.get("WORLD_SIZE", "1")),
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dl, validate_dl)

    # evaluate model
    model.cuda()
    evaluate(
        SongDataset(["train"], **ds_kwargs, frames_per_item=0),
        model,
        "train_ds_evaluation",
        args.frames_per_item,
        logger,
    )
    evaluate(
        SongDataset(["validate"], **ds_kwargs, frames_per_item=0),
        model,
        "validate_ds_evaluation",
        args.frames_per_item,
        logger,
    )


if __name__ == "__main__":
    parser = create_argparser()
    train(parser.parse_args())
