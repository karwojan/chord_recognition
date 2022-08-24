import torch
import torch.distributed as dist
import argparse

# from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm
from einops import rearrange

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
    # init torch distributed
    # dist.init_process_group()

    # init mlflow
    mlflow.set_experiment("btc_custom_implementation")
    mlflow.log_params(args.__dict__)

    # init datasets and data loaders
    ds_kwargs = {
        "sample_rate": args.sample_rate,
        "frame_size": args.frame_size,
        "hop_size": args.hop_size,
        "audio_preprocessing": CQTPreprocessing(),
        "labels_vocabulary": "maj_min",
        "subsets": ["isophonics", "robbie_williams", "uspop"]
    }
    train_ds = SongDataset(["train"], **ds_kwargs, frames_per_item=args.frames_per_item, pitch_shift_augment=args.pitch_shift_augment)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=5
    )
    validate_ds = SongDataset(["validate"], **ds_kwargs, frames_per_item=args.frames_per_item)
    validate_dl = DataLoader(
        validate_ds, batch_size=args.batch_size, shuffle=True, num_workers=5
    )

    # init model and optimizer
    btc = BTC(144, 128, 4, 8, train_ds.n_classes, dropout_p=0.2).cuda()
    # btc = DistributedDataParallel(btc.cuda())
    optimizer = torch.optim.AdamW(btc.parameters())

    for epoch in tqdm(range(args.n_epochs), unit="epoch"):
        # train
        btc.train()
        all_predictions = 0
        all_correct_predictions = 0
        for audio, labels in tqdm(train_dl, unit="batch", total=len(train_dl)):
            audio, labels = audio.cuda(), labels.cuda()
            logits = btc(audio)
            loss = torch.nn.functional.cross_entropy(
                rearrange(logits, "b s c -> b c s"),
                labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predictions = rearrange(torch.argmax(logits, dim=2), "b s -> (b s)")
                labels = rearrange(labels, "b s -> (b s)")
                mlflow.log_metric("train / batch / loss", loss)
                mlflow.log_metric("train / batch / accuracy", (predictions == labels).sum() / len(predictions))

                all_predictions += len(predictions)
                all_correct_predictions += (predictions == labels).sum()
        mlflow.log_metric("train / epoch / accuracy", all_correct_predictions / all_predictions)

        # validate
        btc.eval()
        with torch.no_grad():
            all_predictions = 0
            all_correct_predictions = 0
            for audio, labels in tqdm(
                validate_dl, unit="batch", total=len(validate_dl)
            ):
                audio, labels = audio.cuda(), rearrange(labels.cuda(), "b s -> (b s)")
                predictions = rearrange(torch.argmax(btc(audio), dim=2), "b s -> (b s)")

                all_predictions += len(predictions)
                all_correct_predictions += (predictions == labels).sum()
        mlflow.log_metric("validate / epoch / accuracy", all_correct_predictions / all_predictions)

    # evaluate model
    evaluate(SongDataset(["train"], **ds_kwargs, frames_per_item=0), btc, "train_ds_evaluation", args.frames_per_item)
    evaluate(SongDataset(["validate"], **ds_kwargs, frames_per_item=0), btc, "validate_ds_evaluation", args.frames_per_item)


if __name__ == "__main__":
    parser = create_argparser()
    train(parser.parse_args())
