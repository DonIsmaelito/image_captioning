# Check the caption-to-image mapping with a tiny on-disk COCO-shaped dataset

import json

import h5py
import numpy as np

from captioning.data import load_coco_data, sample_coco_minibatch


def test_loader_and_minibatch_keep_captions_matched_to_images(tmp_path):
    with h5py.File(tmp_path / "coco2014_captions.h5", "w") as file:
        for split in ("train", "val"):
            file[f"{split}_captions"] = [[1, 3, 2], [1, 4, 2], [1, 3, 2]]
            file[f"{split}_image_idxs"] = [1, 0, 1]
    for split in ("train", "val"):
        with h5py.File(tmp_path / f"{split}2014_vgg16_fc7_pca.h5", "w") as file:
            file["features"] = [[10., 11.], [20., 21.]]
        (tmp_path / f"{split}2014_urls.txt").write_text("dog.jpg\ncat.jpg\n")
    vocabulary = ["<NULL>", "<START>", "<END>", "cat", "dog"]
    (tmp_path / "coco2014_vocab.json").write_text(json.dumps({
        "idx_to_word": vocabulary, "word_to_idx": {word: i for i, word in enumerate(vocabulary)},
    }))
    np.random.seed(231)
    data = load_coco_data(tmp_path, max_train=20)
    for split in ("train", "val"):
        captions, features, urls = sample_coco_minibatch(data, 100, split)
        for caption, feature, url in zip(captions, features, urls):
            if caption[1] == 3:
                np.testing.assert_array_equal(feature, [20, 21])
                assert url == "cat.jpg"
            else:
                np.testing.assert_array_equal(feature, [10, 11])
                assert url == "dog.jpg"
