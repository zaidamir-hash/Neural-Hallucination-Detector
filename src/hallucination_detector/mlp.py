"""Frozen-feature MLP training, checkpoint restoration, and validation selection."""
from __future__ import annotations

import copy
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hallucination_detector.baselines.selection import (
    select_threshold, validate_labels, validation_metrics,
)


class HallucinationMLP(nn.Module):
    """Return one raw logit per row; sigmoid is applied only for prediction."""

    def __init__(self, settings: dict[str, Any]):
        super().__init__()
        widths = settings["architecture"]
        drops = settings["dropout"]
        if (widths != [1543, 256, 64, 1] or drops != [0.2, 0.1]
                or settings["activation"] != "ReLU"):
            raise ValueError("Incompatible locked MLP architecture")
        self.layers = nn.Sequential(
            nn.Linear(widths[0], widths[1]), nn.ReLU(), nn.Dropout(drops[0]),
            nn.Linear(widths[1], widths[2]), nn.ReLU(), nn.Dropout(drops[1]),
            nn.Linear(widths[2], widths[3]),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def seed_everything(seed: int) -> None:
    """Seed all RNGs and require deterministic PyTorch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def development_role(split: str) -> None:
    """Reject test inputs before inspecting arrays or opening paths."""
    if split not in ("train", "validation"):
        raise ValueError("Phase 5 accepts only train and validation; test inputs are forbidden")


def make_loader(features: np.ndarray, labels: np.ndarray, split: str,
                batch_size: int, seed: int) -> DataLoader:
    """Shuffle training only, using an independent seeded generator."""
    development_role(split)
    validate_labels(labels)
    if (features.shape != (len(labels), 1543) or not np.isfinite(features).all()):
        raise ValueError("Expected finite aligned 1543-dimensional features")
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32)),
                            torch.from_numpy(labels.astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      generator=generator, num_workers=0, drop_last=False)


def epoch_loss(model: nn.Module, loader: DataLoader, device: str,
               optimizer: torch.optim.Optimizer | None = None) -> float:
    """Aggregate batch-mean BCE by sample count, including the last batch."""
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    total, count = 0.0, 0
    with torch.enable_grad() if training else torch.inference_mode():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite MLP loss")
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
    if count == 0:
        raise ValueError("Empty epoch")
    return total / count


def predict_validation(model: nn.Module, features: np.ndarray, batch_size: int = 128,
                       device: str = "cpu", split: str = "validation") -> np.ndarray:
    """Compute ordered validation probabilities with dropout disabled."""
    if split != "validation":
        raise ValueError("Phase 5 prediction accepts validation only; test inputs are forbidden")
    if features.ndim != 2 or features.shape[1] != 1543 or not np.isfinite(features).all():
        raise ValueError("Expected finite 1543-dimensional validation features")
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start:start+batch_size], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(scores)


def load_checkpoint(path: Path, expected_metadata: dict[str, Any],
                    device: str = "cpu") -> tuple[HallucinationMLP, dict[str, Any]]:
    """Load tensor/primitive-only checkpoints and require exact compatibility."""
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle["compatibility"] != expected_metadata:
        raise ValueError("Incompatible checkpoint schema, preprocessing, scaler, or provenance")
    model = HallucinationMLP(bundle["configuration"]).to(device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    return model, bundle


def train_seed(train: dict[str, Any], validation: dict[str, Any], seed: int,
               settings: dict[str, Any], checkpoint: Path, compatibility: dict[str, Any],
               device: str = "cpu") -> dict[str, Any]:
    """Fit on training, restore minimum-validation-loss weights, then select threshold."""
    if train["split"] != "train" or validation["split"] != "validation":
        raise ValueError("Expected train and validation roles; test inputs are forbidden")
    if settings["loss"] != "BCEWithLogitsLoss" or settings["optimizer"] != "AdamW":
        raise ValueError("Locked loss and optimizer required")
    seed_everything(seed)
    model = HallucinationMLP(settings).to(device)
    train_loader = make_loader(train["features_scaled"], train["labels"], "train", settings["batch_size"], seed)
    validation_loader = make_loader(validation["features_scaled"], validation["labels"], "validation", settings["batch_size"], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, settings["max_epochs"] + 1):
        training_loss = epoch_loss(model, train_loader, device, optimizer)
        validation_loss = epoch_loss(model, validation_loader, device)
        improved = validation_loss < best_loss
        history.append({"epoch": epoch, "training_loss": training_loss,
                        "validation_loss": validation_loss, "improved": improved})
        if improved:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "seed": seed, "best_epoch": best_epoch, "best_validation_loss": best_loss,
                        "configuration": copy.deepcopy(settings), "compatibility": compatibility}, checkpoint)
        else:
            stale += 1
        if stale >= settings["early_stopping_patience"]:
            break
    model, bundle = load_checkpoint(checkpoint, compatibility, device)
    probabilities = predict_validation(model, validation["features_scaled"], settings["batch_size"], device)
    threshold = select_threshold(validation["labels"], probabilities)
    bundle["threshold"] = threshold
    torch.save(bundle, checkpoint)
    return {"seed": seed, "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "epochs_completed": len(history), "stopped_early": len(history) < settings["max_epochs"],
            "threshold": threshold, "validation_metrics": validation_metrics(validation["labels"], probabilities, threshold),
            "history": history}


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Sample SD (ddof=1); median validation F1, with smaller-seed ties."""
    median = float(np.median([r["validation_metrics"]["f1"] for r in runs]))
    representative = min(r["seed"] for r in runs if r["validation_metrics"]["f1"] == median)
    summary = {}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"):
        values = np.asarray([r["validation_metrics"][metric] for r in runs], dtype=float)
        summary[metric] = {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0, ddof=1).tolist()}
    return {"representative_seed": representative, "standard_deviation": "sample standard deviation, ddof=1",
            "validation_metrics": summary}
