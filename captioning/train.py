# Run a 50-caption overfitting experiment locally

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from captioning.trainer import CaptionTrainer, caption_loss
from captioning.model import CaptioningTransformer
from captioning.data import BASE_DIR, decode_captions, load_coco_data


@torch.no_grad()
def evaluate(model, data, split, limit=50):
    # Report teacher-forced loss and unpadded next-word accuracy
    was_training = model.training
    model.eval()
    parameter = next(model.parameters())
    captions = torch.as_tensor(data[f"{split}_captions"][:limit],
                               dtype=torch.long, device=parameter.device)
    rows = data[f"{split}_image_idxs"][:limit]
    features = torch.as_tensor(data[f"{split}_features"][rows],
                               dtype=parameter.dtype, device=parameter.device)
    try:
        logits = model(features, captions[:, :-1])
        targets = captions[:, 1:]
        mask = targets != model._null
        accuracy = ((logits.argmax(-1) == targets) & mask).sum() / mask.sum().clamp_min(1)
        return {"loss": caption_loss(logits, targets, model._null).item(),
                "token_accuracy": accuracy.item(), "caption_count": len(captions)}
    finally:
        model.train(was_training)


def caption_examples(model, data, split, count=3):
    rows = data[f"{split}_image_idxs"][:count]
    generated = model.sample(data[f"{split}_features"][rows], max_length=model.max_length)
    predicted = decode_captions(generated, data["idx_to_word"])
    reference = decode_captions(data[f"{split}_captions"][:count], data["idx_to_word"])
    return [{"prediction": pred, "reference": ref, "image_url": str(url)}
            for pred, ref, url in zip(predicted, reference, data[f"{split}_urls"][rows])]


def main():
    parser = argparse.ArgumentParser(description="Run a 50-caption overfitting experiment locally.")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output", type=Path, default=Path("runs/coco"))
    parser.add_argument("--max-train", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--require-loss", type=float, help="Fail if final training loss is not below this")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_coco_data(args.data_dir, max_train=args.max_train)
    config = dict(input_dim=data["train_features"].shape[1], wordvec_dim=256,
                  num_heads=2, num_layers=2, max_length=30)
    model = CaptioningTransformer(data["word_to_idx"], **config).to(args.device)
    solver = CaptionTrainer(
        model, data, num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    start = time.monotonic()
    solver.train()
    training_seconds = time.monotonic() - start
    metrics = {
        "seed": args.seed, "device": args.device, "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "training_seconds": training_seconds,
        "max_train": args.max_train, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "threads": args.threads,
        "first_training_loss": solver.loss_history[0],
        "final_training_loss": solver.loss_history[-1],
        "training_target": 0.05,
        "meets_training_target": solver.loss_history[-1] < 0.05,
        "train_eval": evaluate(model, data, "train"),
        "validation_eval": evaluate(model, data, "val"),
        "loss_history": solver.loss_history,
        "examples": {split: caption_examples(model, data, split) for split in ("train", "val")},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": config,
                "word_to_idx": data["word_to_idx"], "idx_to_word": data["idx_to_word"]},
               args.output / "model.pt")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Final training loss: {metrics['final_training_loss']:.6f}")
    print(f"Validation loss on 50 captions: {metrics['validation_eval']['loss']:.6f}")
    print(f"Saved model and metrics to {args.output}")
    for split, examples in metrics["examples"].items():
        print(f"\n{split.upper()} examples:")
        for example in examples:
            print(f"  Predicted: {example['prediction']}\n  Reference: {example['reference']}")
    if args.require_loss is not None and metrics["final_training_loss"] >= args.require_loss:
        raise SystemExit(f"Training loss did not reach {args.require_loss}")


if __name__ == "__main__":
    main()
