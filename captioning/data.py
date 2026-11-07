# Load prepared COCO features and captions (no CNN needed)

# Caption rows map to image rows through
# train_image_idxs and val_image_idxs since one image can have several captions

import json
from pathlib import Path

import h5py
import numpy as np

BASE_DIR = Path("data") / "coco_captioning"


def load_coco_data(base_dir=BASE_DIR, max_train=None, pca_features=True):
    base_dir = Path(base_dir)
    if not (base_dir / "coco2014_captions.h5").exists():
        raise FileNotFoundError("COCO data missing. Run: python -m captioning.prepare_data --archive /path/to/coco_captioning.zip")
    data = {}
    with h5py.File(base_dir / "coco2014_captions.h5", "r") as file:
        for name in file:
            data[name] = np.asarray(file[name])
    suffix = "_pca" if pca_features else ""
    for split in ("train", "val"):
        with h5py.File(base_dir / f"{split}2014_vgg16_fc7{suffix}.h5", "r") as file:
            data[f"{split}_features"] = np.asarray(file["features"], dtype=np.float32)
        data[f"{split}_urls"] = np.array(
            (base_dir / f"{split}2014_urls.txt").read_text().splitlines()
        )
    data.update(json.loads((base_dir / "coco2014_vocab.json").read_text()))
    if max_train is not None:
        if max_train < 1:
            raise ValueError("max_train must be positive")
        # Use seeded selection, including sampling with replacement
        rows = np.random.randint(len(data["train_captions"]), size=max_train)
        data["train_captions"] = data["train_captions"][rows]
        data["train_image_idxs"] = data["train_image_idxs"][rows]
    return data


def sample_coco_minibatch(data, batch_size=100, split="train"):
    rows = np.random.choice(len(data[f"{split}_captions"]), batch_size)
    captions = data[f"{split}_captions"][rows]
    image_rows = data[f"{split}_image_idxs"][rows]
    features = data[f"{split}_features"][image_rows]
    urls = data[f"{split}_urls"][image_rows]
    return captions, features, urls


def decode_captions(captions, idx_to_word):
    # Convert token IDs to readable text, hiding the control tokens
    captions = np.asarray(captions)
    single = captions.ndim == 1
    if single:
        captions = captions[None]
    sentences = []
    for caption in captions:
        words = []
        for token in caption:
            word = idx_to_word[int(token)]
            if word == "<END>":
                break
            if word not in ("<NULL>", "<START>"):
                words.append(word)
        sentences.append(" ".join(words))
    return sentences[0] if single else sentences
