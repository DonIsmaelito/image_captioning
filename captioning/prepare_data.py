# Prepare a COCO captioning archive; keep only captions and PCA features

import argparse
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

DEFAULT_DEST = Path("data") / "coco_captioning"
FILES = (
    "coco2014_captions.h5", "coco2014_vocab.json",
    "train2014_vgg16_fc7_pca.h5", "val2014_vgg16_fc7_pca.h5",
    "train2014_urls.txt", "val2014_urls.txt",
)


def extract_data(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for name in FILES:
            # Explicit paths prevent archive path traversal and skip unused features
            member = f"coco_captioning/{name}"
            pending = destination / f"{name}.partial"
            with zipped.open(member) as source, pending.open("wb") as output:
                shutil.copyfileobj(source, output)
            pending.replace(destination / name)
            print(f"Extracted {name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare a COCO captioning archive; keep only captions and PCA features.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Use an existing COCO captioning ZIP")
    source.add_argument("--url", help="Download a prepared COCO captioning ZIP from this URL")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    if all((args.destination / name).is_file() for name in FILES):
        print(f"Data already available at {args.destination}")
        return
    if args.archive:
        extract_data(args.archive, args.destination)
    else:
        if not args.url:
            parser.error("Provide --archive /path/to/coco_captioning.zip or --url DATASET_ZIP_URL")
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "coco.zip"
            print("Downloading the dataset archive; this may take a few minutes.", flush=True)
            urllib.request.urlretrieve(args.url, archive)
            extract_data(archive, args.destination)


if __name__ == "__main__":
    main()
