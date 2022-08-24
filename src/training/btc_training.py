import argparse

from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm
from einops import rearrange
import pytorch_lightning as pl

from src.training.dataset import SongDataset
from src.training.preprocessing import CQTPreprocessing
from src.training.btc import BTC
from src.training.btc_evaluate import evaluate


def create_argparser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--frame_size", type=int, required=True)
    parser.add_argument("--hop_size", type=int, required=True)
    parser.add_argument("--frames_per_item", type=int, required=True)
    parser.add_argument("--pitch_shift_augment", action="store_true")

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
    btc = BTC(144, 128, 4, 8, train_ds.n_classes, dropout_p=0.2)

    # train model
    logger = pl.loggers.MLFlowLogger(experiment_name="btc_custom_implementation")
    logger.log_hyperparams(args)
    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(btc, train_dl, validate_dl)

    # evaluate model
    evaluate(
        SongDataset(["train"], **ds_kwargs, frames_per_item=0),
        btc,
        "train_ds_evaluation",
        args.frames_per_item,
    )
    evaluate(
        SongDataset(["validate"], **ds_kwargs, frames_per_item=0),
        btc,
        "validate_ds_evaluation",
        args.frames_per_item,
    )


if __name__ == "__main__":
    parser = create_argparser()
    train(parser.parse_args())
