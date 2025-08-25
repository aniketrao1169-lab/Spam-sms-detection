from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split


LABEL_COLUMN_CANDIDATES = ["label", "category", "v1"]
TEXT_COLUMN_CANDIDATES = ["text", "message", "v2"]


def _infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
	"""Infer label and text column names from common variants."""
	label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in df.columns.str.lower()), None)
	text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in df.columns.str.lower()), None)
	if label_col is None or text_col is None:
		raise ValueError(
			"Could not infer label/text columns. Expected columns like: "
			f"{LABEL_COLUMN_CANDIDATES} for labels and {TEXT_COLUMN_CANDIDATES} for text."
		)
	return label_col, text_col


def load_dataset(csv_path: str) -> pd.DataFrame:
	"""Load the dataset and normalize to columns: `label`, `text`.

	Tries multiple encodings for robustness (utf-8, latin-1, cp1252).
	Supports Kaggle SMS Spam dataset with columns `v1` (label) and `v2` (text).
	"""
	df = None
	for enc in ("utf-8", "latin-1", "cp1252"):
		try:
			df = pd.read_csv(csv_path, encoding=enc)
			break
		except UnicodeDecodeError:
			continue
	if df is None:
		# Fallback: ignore undecodable bytes
		df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="ignore")
	# Normalize column names to lowercase for matching
	df.columns = [c.lower() for c in df.columns]
	label_col, text_col = _infer_columns(df)
	# Select and rename
	df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
	# Clean labels to canonical 'ham'/'spam'
	df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ham": "ham", "spam": "spam"})
	# Drop rows with non-standard labels if any
	df = df.dropna(subset=["label", "text"])
	return df


def clean_text(text: str) -> str:
	"""Lowercase, remove punctuation and extra spaces, remove stopwords."""
	if not isinstance(text, str):
		return ""
	text = text.lower()
	# Replace URLs/emails with space and strip non-letters
	text = re.sub(r"https?://\S+|www\.\S+", " ", text)
	text = re.sub(r"\S+@\S+", " ", text)
	text = re.sub(r"[^a-z\s]", " ", text)
	# Collapse whitespace
	words = text.split()
	filtered = [w for w in words if w not in ENGLISH_STOP_WORDS]
	return " ".join(filtered)


@dataclass
class VectorizedData:
	X_train: any
	X_test: any
	y_train: pd.Series
	y_test: pd.Series
	vectorizer: TfidfVectorizer


def vectorize_and_split(
	df: pd.DataFrame,
	test_size: float = 0.2,
	random_state: int = 42,
) -> VectorizedData:
	"""Clean text, split, fit TF-IDF on train, transform both train/test."""
	texts = df["text"].astype(str).apply(clean_text)
	labels = df["label"].astype(str)
	X_train_text, X_test_text, y_train, y_test = train_test_split(
		texts, labels, test_size=test_size, random_state=random_state, stratify=labels
	)
	vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, lowercase=True)
	X_train = vectorizer.fit_transform(X_train_text)
	X_test = vectorizer.transform(X_test_text)
	return VectorizedData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, vectorizer=vectorizer)


