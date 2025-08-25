from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from src.preprocessing import load_dataset, vectorize_and_split
from src.models import train_models, select_best_model, save_artifacts, load_artifacts
from src.evaluate import compute_metrics, print_metrics, save_confusion_matrix, save_metrics
from src.utils import project_paths, ensure_dir
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def train_and_evaluate(dataset_path: str) -> Dict[str, float]:
	console = Console()
	paths = project_paths()
	console.print(Panel.fit("[bold cyan]Spam SMS Detection[/bold cyan]", border_style="cyan"))
	console.print("[bold]Step 1/5:[/bold] Loading dataset :white_check_mark:")
	df = load_dataset(dataset_path)
	console.print("[bold]Step 2/5:[/bold] Preprocessing & vectorizing :white_check_mark:")
	vec_data = vectorize_and_split(df)
	console.print("[bold]Step 3/5:[/bold] Training models :white_check_mark:")
	models = train_models(vec_data.X_train, vec_data.y_train)
	console.print("[bold]Step 4/5:[/bold] Evaluating models :white_check_mark:")
	all_scores: Dict[str, float] = {}
	best_name, best_model, f1_scores = select_best_model(models, vec_data.X_test, vec_data.y_test)
	# Render metrics table
	table = Table(title="Model Metrics", title_style="bold magenta")
	table.add_column("Model", justify="left", style="cyan", no_wrap=True)
	table.add_column("Accuracy", justify="right")
	table.add_column("Precision", justify="right")
	table.add_column("Recall", justify="right")
	table.add_column("F1", justify="right")
	for name, model in models.items():
		pred = model.predict(vec_data.X_test)
		metrics = compute_metrics(vec_data.y_test, pred)
		table.add_row(name, f"{metrics['accuracy']:.3f}", f"{metrics['precision']:.3f}", f"{metrics['recall']:.3f}", f"{metrics['f1']:.3f}")
		all_scores[name] = metrics["f1"]
	console.print(table)
	# Save confusion matrix and metrics for best model
	best_pred = best_model.predict(vec_data.X_test)
	ensure_dir(paths["results"]) 
	cm_path = save_confusion_matrix(vec_data.y_test, best_pred, Path(paths["results"]) / "confusion_matrix.png")
	metrics_path = save_metrics(compute_metrics(vec_data.y_test, best_pred), Path(paths["results"]) / "metrics.json")
	console.print(f"Best model: [bold green]{best_name}[/bold green]")
	console.print(f"Confusion matrix: [italic]{cm_path}[/italic]")
	console.print(f"Metrics JSON: [italic]{metrics_path}[/italic]")
	console.print("[bold]Step 5/5:[/bold] Saving artifacts :white_check_mark:")
	save_artifacts(best_model, vec_data.vectorizer, paths["models"])
	return {"best_model": best_name, **{f"f1_{k}": v for k, v in all_scores.items()}}


def predict_message(message: str, dataset_path: str | None = None) -> str:
	paths = project_paths()
	model, vectorizer = load_artifacts(paths["models"])
	if model is None or vectorizer is None:
		if dataset_path is None:
			raise FileNotFoundError(
				"No trained model found. Provide --dataset to train before prediction."
			)
		print("No existing model found. Training now...")
		train_and_evaluate(dataset_path)
		model, vectorizer = load_artifacts(paths["models"])  # reload
	
	X = vectorizer.transform([message])
	pred = model.predict(X)[0]
	return "SPAM" if pred == "spam" else "NOT SPAM"


def main():
	parser = argparse.ArgumentParser(description="Spam SMS Detection (CLI)")
	parser.add_argument("--dataset", type=str, default="data/spam.csv", help="Path to dataset CSV")
	parser.add_argument("--message", type=str, default=None, help="Custom SMS message to classify")
	args = parser.parse_args()

	console = Console()
	if args.message:
		label = predict_message(args.message, args.dataset)
		color = "red" if label == "SPAM" else "green"
		console.print(Panel.fit(f"Prediction: [bold {color}]{label}[/bold {color}]", border_style=color))
	else:
		train_and_evaluate(args.dataset)


if __name__ == "__main__":
	main()


