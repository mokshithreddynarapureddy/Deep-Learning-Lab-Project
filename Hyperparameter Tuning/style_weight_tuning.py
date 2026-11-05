from pathlib import Path
import re
import string
import time

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = ROOT / "data" / "train.csv"

N_SPLITS = 5
RANDOM_STATE = 42

# Current champion uses 0.25.
# Test weaker and stronger stylistic feature contributions.
STYLE_WEIGHTS = [0.05, 0.10, 0.25, 0.50, 1.00]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRAIN_PATH)

X = df["text"].fillna("").astype(str)
y = df["model"]
groups = df["prompt_id"]

print("=" * 75)
print("NAME THAT LLM - STYLE WEIGHT TUNING")
print("=" * 75)

print(f"Training examples : {len(df)}")
print(f"Classes           : {y.nunique()}")
print(f"Unique prompts    : {groups.nunique()}")
print(f"Validation folds  : {N_SPLITS}")
print(f"Style weights     : {STYLE_WEIGHTS}")
print()


# ============================================================
# EXTRACT STYLISTIC FEATURES
# ============================================================

def extract_style_features(text):
    text = str(text)

    chars = len(text)
    words = re.findall(r"\b\w+\b", text)

    sentences = [
        s for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]

    paragraphs = [
        p for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    word_lengths = [len(word) for word in words]

    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)

    uppercase_count = sum(c.isupper() for c in text)
    lowercase_count = sum(c.islower() for c in text)
    digit_count = sum(c.isdigit() for c in text)
    whitespace_count = sum(c.isspace() for c in text)

    punctuation_count = sum(
        c in string.punctuation for c in text
    )

    markdown_headings = len(
        re.findall(r"(?m)^\s{0,3}#{1,6}\s", text)
    )

    bullet_points = len(
        re.findall(r"(?m)^\s*[-*+]\s+", text)
    )

    numbered_items = len(
        re.findall(r"(?m)^\s*\d+[.)]\s+", text)
    )

    code_fences = text.count("```")
    inline_code = text.count("`") - 3 * code_fences

    urls = len(
        re.findall(r"https?://\S+|www\.\S+", text)
    )

    return [
        chars,
        word_count,
        sentence_count,
        paragraph_count,
        text.count("\n"),

        chars / max(word_count, 1),
        word_count / max(sentence_count, 1),
        sentence_count / max(paragraph_count, 1),

        np.mean(word_lengths) if word_lengths else 0,
        np.std(word_lengths) if word_lengths else 0,
        max(word_lengths) if word_lengths else 0,

        uppercase_count / max(chars, 1),
        lowercase_count / max(chars, 1),
        digit_count / max(chars, 1),
        whitespace_count / max(chars, 1),
        punctuation_count / max(chars, 1),

        text.count("!") / max(word_count, 1),
        text.count("?") / max(word_count, 1),
        text.count(",") / max(word_count, 1),
        text.count(";") / max(word_count, 1),
        text.count(":") / max(word_count, 1),
        (text.count('"') + text.count("'")) / max(word_count, 1),

        markdown_headings,
        bullet_points,
        numbered_items,
        code_fences,
        inline_code,
        urls,
    ]


print("Extracting stylistic features...")

style_array = np.asarray(
    [extract_style_features(text) for text in X],
    dtype=np.float64,
)

print("Stylistic feature matrix:", style_array.shape)
print()


# ============================================================
# PREPARE CROSS-VALIDATION
# ============================================================

cv = GroupKFold(n_splits=N_SPLITS)

# Store one out-of-fold prediction per training example.
oof_predictions = {
    weight: np.empty(len(df), dtype=object)
    for weight in STYLE_WEIGHTS
}

oof_baseline = np.empty(len(df), dtype=object)

fold_scores = {
    weight: []
    for weight in STYLE_WEIGHTS
}

baseline_fold_scores = []

start_all = time.time()


# ============================================================
# CROSS-VALIDATION LOOP
# ============================================================

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y, groups=groups),
    start=1,
):
    fold_start = time.time()

    print("=" * 75)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 75)

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # --------------------------------------------------------
    # Fit character TF-IDF using training data only.
    # --------------------------------------------------------

    print("Fitting character TF-IDF...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=250_000,
        sublinear_tf=True,
        max_df=0.98,
        dtype=np.float32,
    )

    X_char_train = vectorizer.fit_transform(X_train)
    X_char_valid = vectorizer.transform(X_valid)

    # --------------------------------------------------------
    # Baseline classifier.
    # --------------------------------------------------------

    baseline_classifier = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )

    baseline_classifier.fit(X_char_train, y_train)

    baseline_predictions = baseline_classifier.predict(
        X_char_valid
    )

    oof_baseline[valid_idx] = baseline_predictions

    baseline_score = balanced_accuracy_score(
        y_valid, baseline_predictions
    )

    baseline_fold_scores.append(baseline_score)

    print(f"Character-only baseline: {baseline_score:.5f}")

    # --------------------------------------------------------
    # Scale style features using training data only.
    # --------------------------------------------------------

    scaler = StandardScaler()

    style_train = scaler.fit_transform(
        style_array[train_idx]
    )

    style_valid = scaler.transform(
        style_array[valid_idx]
    )

    style_train_sparse = csr_matrix(
        style_train.astype(np.float32)
    )

    style_valid_sparse = csr_matrix(
        style_valid.astype(np.float32)
    )

    # Keep the same normalization strategy as the previous
    # experiment. Only the STYLE_WEIGHT will change.
    style_train_sparse = normalize(
        style_train_sparse, norm="l2"
    )

    style_valid_sparse = normalize(
        style_valid_sparse, norm="l2"
    )

    # --------------------------------------------------------
    # Evaluate each style weight.
    # --------------------------------------------------------

    for weight in STYLE_WEIGHTS:

        X_combined_train = hstack(
            [
                X_char_train,
                weight * style_train_sparse,
            ],
            format="csr",
        )

        X_combined_valid = hstack(
            [
                X_char_valid,
                weight * style_valid_sparse,
            ],
            format="csr",
        )

        classifier = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000,
        )

        classifier.fit(X_combined_train, y_train)

        predictions = classifier.predict(
            X_combined_valid
        )

        oof_predictions[weight][valid_idx] = predictions

        score = balanced_accuracy_score(
            y_valid, predictions
        )

        fold_scores[weight].append(score)

        print(
            f"Style weight {weight:>4.2f}: "
            f"balanced accuracy = {score:.5f}"
        )

    print(
        f"Fold execution time: "
        f"{time.time() - fold_start:.1f} seconds"
    )
    print()


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 75)
print("FINAL STYLE WEIGHT RESULTS")
print("=" * 75)

results = []

baseline_oof_score = balanced_accuracy_score(
    y, oof_baseline
)

results.append({
    "configuration": "character_only",
    "style_weight": 0.0,
    "oof_balanced_accuracy": baseline_oof_score,
    "mean_fold_accuracy": np.mean(baseline_fold_scores),
    "fold_std": np.std(baseline_fold_scores),
})

for weight in STYLE_WEIGHTS:

    predictions = oof_predictions[weight]

    if pd.isna(predictions).any():
        raise ValueError(
            f"Missing predictions for style weight {weight}"
        )

    overall_score = balanced_accuracy_score(
        y, predictions
    )

    results.append({
        "configuration": f"character_plus_style_{weight:.2f}",
        "style_weight": weight,
        "oof_balanced_accuracy": overall_score,
        "mean_fold_accuracy": np.mean(fold_scores[weight]),
        "fold_std": np.std(fold_scores[weight]),
    })


results_df = pd.DataFrame(results).sort_values(
    "oof_balanced_accuracy",
    ascending=False,
)

print(results_df.to_string(index=False))

results_path = OUTPUT_DIR / "style_weight_results.csv"

results_df.to_csv(results_path, index=False)


# ============================================================
# SAVE OOF PREDICTIONS
# ============================================================

prediction_data = {
    "id": df["id"],
    "prompt_id": df["prompt_id"],
    "actual_model": y,
    "character_only": oof_baseline,
}

for weight in STYLE_WEIGHTS:
    prediction_data[f"style_weight_{weight:.2f}"] = (
        oof_predictions[weight]
    )

predictions_df = pd.DataFrame(prediction_data)

predictions_path = OUTPUT_DIR / "style_weight_oof_predictions.csv"

predictions_df.to_csv(predictions_path, index=False)


# ============================================================
# SUMMARY
# ============================================================

best_row = results_df.iloc[0]

print()
print("=" * 75)
print("EXPERIMENT COMPLETE")
print("=" * 75)

print("Best configuration:", best_row["configuration"])
print(
    "Best OOF balanced accuracy:",
    f"{best_row['oof_balanced_accuracy']:.5f}",
)

print(f"Results saved: {results_path}")
print(f"Predictions saved: {predictions_path}")

print(
    f"Total execution time: "
    f"{(time.time() - start_all) / 60:.2f} minutes"
)