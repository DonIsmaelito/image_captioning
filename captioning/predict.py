# Load a trained checkpoint and caption a prepared COCO image feature

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from captioning.model import CaptioningTransformer
from captioning.data import BASE_DIR, decode_captions


def main():
    parser = argparse.ArgumentParser(description="Load a trained checkpoint and caption a prepared COCO image feature.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/coco/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--image-index", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CaptioningTransformer(checkpoint["word_to_idx"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    feature_path = args.data_dir / f"{args.split}2014_vgg16_fc7_pca.h5"
    with h5py.File(feature_path, "r") as file:
        if not 0 <= args.image_index < len(file["features"]):
            parser.error("image-index is outside the dataset")
        features = np.asarray(file["features"][args.image_index:args.image_index + 1])
    captions = model.sample(features, max_length=model.max_length)
    print(decode_captions(captions, checkpoint["idx_to_word"])[0])
    urls = (args.data_dir / f"{args.split}2014_urls.txt").read_text().splitlines()
    print(f"Image: {urls[args.image_index]}")


if __name__ == "__main__":
    main()
