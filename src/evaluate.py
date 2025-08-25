from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	precision_score,
	recall_score,
	f1_score,
)

from .utils import ensure_dir, save_json


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
	"""Return accuracy, precision, recall, f1 with positive label 'spam'."""
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, pos_label="spam")),
		"recall": float(recall_score(y_true, y_pred, pos_label="spam")),
		"f1": float(f1_score(y_true, y_pred, pos_label="spam")),
	}


def print_metrics(metrics: Dict[str, float], model_name: str) -> None:
	acc = metrics["accuracy"]; pre = metrics["precision"]; rec = metrics["recall"]; f1 = metrics["f1"]
	print(f"  - {model_name:<10} | Acc: {acc:.3f}  P: {pre:.3f}  R: {rec:.3f}  F1: {f1:.3f}")


def save_confusion_matrix(y_true, y_pred, out_path: str | Path) -> Path:
	"""Save a confusion matrix heatmap as PNG."""
	cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
	plt.figure(figsize=(5, 4), dpi=140)
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	ensure_dir(Path(out_path).parent)
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()
	return Path(out_path)


def save_metrics(metrics: Dict[str, float], out_path: str | Path) -> Path:
	ensure_dir(Path(out_path).parent)
	save_json(metrics, out_path)
	return Path(out_path)


