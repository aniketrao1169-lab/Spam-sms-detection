from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

from .utils import ensure_dir


def train_models(X_train, y_train) -> Dict[str, object]:
	"""Train multiple classifiers and return a dict of name -> fitted model."""
	models: Dict[str, object] = {}
	models["NaiveBayes"] = MultinomialNB().fit(X_train, y_train)
	models["LogReg"] = LogisticRegression(max_iter=2000, n_jobs=None).fit(X_train, y_train)
	models["LinearSVM"] = LinearSVC().fit(X_train, y_train)
	return models


def select_best_model(models: Dict[str, object], X_val, y_val) -> Tuple[str, object, Dict[str, float]]:
	"""Select the best model based on F1 score (positive class = 'spam')."""
	scores: Dict[str, float] = {}
	for name, model in models.items():
		pred = model.predict(X_val)
		scores[name] = f1_score(y_val, pred, pos_label="spam")
	best_name = max(scores, key=scores.get)
	return best_name, models[best_name], scores


def save_artifacts(model, vectorizer, output_dir: str | Path) -> Tuple[Path, Path]:
	"""Persist the model and vectorizer to disk."""
	output_dir = Path(output_dir)
	ensure_dir(output_dir)
	model_path = output_dir / "best_model.joblib"
	vec_path = output_dir / "vectorizer.joblib"
	joblib.dump(model, model_path)
	joblib.dump(vectorizer, vec_path)
	return model_path, vec_path


def load_artifacts(model_dir: str | Path) -> Tuple[object | None, object | None]:
	"""Load model and vectorizer if available; otherwise return (None, None)."""
	model_dir = Path(model_dir)
	model_path = model_dir / "best_model.joblib"
	vec_path = model_dir / "vectorizer.joblib"
	if model_path.exists() and vec_path.exists():
		model = joblib.load(model_path)
		vectorizer = joblib.load(vec_path)
		return model, vectorizer
	return None, None


