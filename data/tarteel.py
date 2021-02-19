import argparse
from itertools import repeat
from multiprocessing import Pool
import os
from pathlib import Path
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest

parser = argparse.ArgumentParser(description="Process Tarteel Dataset")
parser = add_data_opts(parser)
parser.add_argument("--target-dir", default="everyayah_dataset_wav/", type=str,
                    help="Directory to store the dataset.")
parser.add_argument("--txt-dir", default="txt", type=str, help="Directory of labels.")
parser.add_argument("--val-fraction", default=0.1, type=float,
                    help="Number of files in the training set to use as validation.")
parser.add_argument("-e", "--extension", default="mp3", type=str, help="File extension")
parser.add_argument("--seed", default=0, type=int, help="Seed for train/test/val split.")
args = parser.parse_args()


def _move_file(old_new_paths):
    file_path, transcript_path, new_wav_dir, new_transcript_dir = old_new_paths
    new_file_path = new_wav_dir / Path(file_path).name
    new_transcript_path = new_transcript_dir / Path(transcript_path).name
    os.symlink(Path(file_path).absolute(), new_file_path.absolute())
    os.symlink(Path(transcript_path).absolute(), new_transcript_path.absolute())


def _save_wav_transcripts(data_type: str,
                          file_paths,
                          transcript_paths,
                          target_dir: Path,
                          num_workers: int = 8,
                          file_extension: str = "wav"):
    data_path = Path(target_dir) / data_type
    if data_path.exists():
        shutil.rmtree(data_path)

    new_transcript_dir = data_path / args.txt_dir
    new_wav_dir = data_path / file_extension
    new_transcript_dir.mkdir(parents=True)
    new_wav_dir.mkdir(parents=True)

    copy_tuple = zip(file_paths, transcript_paths, repeat(new_wav_dir), repeat(new_transcript_dir))

    if num_workers <= 1:
        for t in tqdm(copy_tuple, desc=f"Creating {data_type} directory", total=len(file_paths)):
            _move_file(t)
    else:
        with Pool(processes=num_workers) as p:
            list(tqdm(p.imap(_move_file, copy_tuple), desc=f"Creating {data_type} directory", total=len(file_paths)))


def _format_training_data(root_path: Path,
                          target_dir: Path,
                          val_fraction: float = 0.10,
                          test_fraction: float = 0.05,
                          seed: int = 0,
                          num_workers: int = 8,
                          file_extension: str = "wav"):
    wav_path = Path(root_path) / file_extension
    txt_path = Path(root_path) / args.txt_dir
    transcripts_files = list(txt_path.glob(f"*.txt"))
    wav_files = [wav_path / f.with_suffix(f".{file_extension}").name for f in transcripts_files]
    # Reformat for multithreading
    wav_files = [f.as_posix() for f in wav_files]
    transcripts_files = [f.as_posix() for f in transcripts_files]

    train_fraction = 1 - val_fraction - test_fraction
    if abs(sum([train_fraction, test_fraction, val_fraction]) - 1.0) > 1e-6:
        raise Exception("Train-test-validation fractions do not sum to 1.")

    split1 = train_fraction + val_fraction
    split2 = 1.0 - (val_fraction / split1)
    train_val_waves, test_waves, train_val_transcripts, test_transcripts = \
        train_test_split(wav_files, transcripts_files, train_size=split1, shuffle=False, random_state=seed)
    train_waves, val_waves, train_transcripts, val_transcripts = \
        train_test_split(train_val_waves, train_val_transcripts, train_size=split2, shuffle=False, random_state=seed)
    print(f"Splits created\nTrain: {len(train_waves)}\nValidation: {len(val_waves)}\nTest: {len(test_waves)}")

    _save_wav_transcripts(data_type="train",
                          file_paths=train_waves,
                          transcript_paths=train_transcripts,
                          target_dir=target_dir,
                          num_workers=num_workers,
                          file_extension=file_extension)

    _save_wav_transcripts(data_type="val",
                          file_paths=val_waves,
                          transcript_paths=val_transcripts,
                          target_dir=target_dir,
                          num_workers=num_workers,
                          file_extension=file_extension)

    _save_wav_transcripts(data_type="test",
                          file_paths=test_waves,
                          transcript_paths=test_transcripts,
                          target_dir=target_dir,
                          num_workers=num_workers,
                          file_extension=file_extension)


def main():
    train_path = Path(args.target_dir) / "train"
    val_path = Path(args.target_dir) / "val"
    test_path = Path(args.target_dir) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    _format_training_data(root_path=args.target_dir,
                          val_fraction=args.val_fraction,
                          target_dir=args.target_dir,
                          seed=args.seed,
                          num_workers=args.num_workers,
                          file_extension=args.extension)

    create_manifest(
        data_path=train_path,
        output_name="train_manifest.json",
        manifest_path=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers,
        file_extension=args.extension
    )

    create_manifest(
        data_path=val_path,
        output_name="val_manifest.json",
        manifest_path=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers,
        file_extension=args.extension
    )

    create_manifest(
        data_path=test_path,
        output_name="test_manifest.json",
        manifest_path=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers,
        file_extension=args.extension
    )


if __name__ == "__main__":
    main()
