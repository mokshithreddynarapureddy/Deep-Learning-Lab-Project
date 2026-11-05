from pathlib import Path
import re
import string
import time
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = ROOT / "data" / "train.csv"

N_SPLITS = 5
RANDOM_STATE = 42

# Best weight from the previous tuning experiment
STYLE_WEIGHT = 0.50

# C values to test
C_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRAIN_PATH)

X = df["text"].fillna("").astype(str)
y = df["model"]
groups = df["prompt_id"]

print("=" * 75)
print("NAME THAT LLM - FAIR LINEARSVC C TUNING")
print("=" * 75)

print(f"Training examples : {len(df)}")
print(f"Classes           : {y.nunique()}")
print(f"Unique prompts    : {groups.nunique()}")
print(f"Validation folds  : {N_SPLITS}")
print(f"Style features    : 28")
print(f"Style weight      : {STYLE_WEIGHT}")
print(f"C values          : {C_VALUES}")
print()


# ============================================================
# ORIGINAL 28 STYLISTIC FEATURES
# Copied from stylistic_features.py
# ============================================================

def extract_style_features(text):
    """Extract the original 28 numerical writing-style features."""

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

    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)

    word_lengths = [len(w) for w in words]

    uppercase_count = sum(c.isupper() for c in text)
    lowercase_count = sum(c.islower() for c in text)
    digit_count = sum(c.isdigit() for c in text)
    whitespace_count = sum(c.isspace() for c in text)

    punctuation_count = sum(
        c in string.punctuation for c in text
    )

    exclamation_count = text.count("!")
    question_count = text.count("?")
    comma_count = text.count(",")
    semicolon_count = text.count(";")
    colon_count = text.count(":")
    quote_count = text.count('"') + text.count("'")

    newline_count = text.count("\n")

    markdown_heading_count = len(
        re.findall(r"(?m)^\s{0,3}#{1,6}\s", text)
    )

    bullet_count = len(
        re.findall(r"(?m)^\s*[-*+]\s+", text)
    )

    numbered_list_count = len(
        re.findall(r"(?m)^\s*\d+[.)]\s+", text)
    )

    code_fence_count = text.count("```")

    inline_code_count = text.count("`") - code_fence_count * 3

    url_count = len(
        re.findall(r"https?://\S+|www\.\S+", text)
    )

    return [
        # 1-5: Response length
        chars,
        word_count,
        sentence_count,
        paragraph_count,
        newline_count,

        # 6-11: Average lengths
        chars / max(word_count, 1),
        word_count / max(sentence_count, 1),
        sentence_count / max(paragraph_count, 1),
        np.mean(word_lengths) if word_lengths else 0,
        np.std(word_lengths) if word_lengths else 0,
        max(word_lengths) if word_lengths else 0,

        # 12-17: Character composition
        uppercase_count / max(chars, 1),
        lowercase_count / max(chars, 1),
        digit_count / max(chars, 1),
        whitespace_count / max(chars, 1),
        punctuation_count / max(chars, 1),

        # 18-23: Punctuation style
        exclamation_count / max(word_count, 1),
        question_count / max(word_count, 1),
        comma_count / max(word_count, 1),
        semicolon_count / max(word_count, 1),
        colon_count / max(word_count, 1),
        quote_count / max(word_count, 1),

        # 24-28: Formatting
        markdown_heading_count,
        bullet_count,
        numbered_list_count,
        code_fence_count,
        inline_code_count,
        url_count,
    ]


print("Extracting original stylistic features...")

style_array = np.array(
    [extract_style_features(text) for text in X],
    dtype=np.float64
)

assert style_array.shape[1] == 28, (
    f"Expected 28 stylistic features, got {style_array.shape[1]}"
)

print(f"Style feature matrix: {style_array.shape}")
print()


# ============================================================
# GROUPED CROSS-VALIDATION
# ============================================================

cv = GroupKFold(n_splits=N_SPLITS)

fold_results = []

oof_predictions = {
    c_value: np.empty(len(df), dtype=object)
    for c_value in C_VALUES
}

start_all = time.time()


# ============================================================
# RUN EXPERIMENT
# ============================================================

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y, groups=groups),
    start=1
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
    # CHARACTER TF-IDF
    # --------------------------------------------------------

    print("Fitting character TF-IDF...")

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=250_000,
        sublinear_tf=True,
        max_df=0.98,
        dtype=np.float32
    )

    X_char_train = char_vectorizer.fit_transform(X_train)
    X_char_valid = char_vectorizer.transform(X_valid)

    # --------------------------------------------------------
    # STYLE FEATURES
    # --------------------------------------------------------

    print("Scaling stylistic features...")

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

    # Match the original feature preprocessing
    style_train_sparse = normalize(
        style_train_sparse,
        norm="l2"
    )

    style_valid_sparse = normalize(
        style_valid_sparse,
        norm="l2"
    )

    # Combine character and style features
    X_combined_train = hstack(
        [
            X_char_train,
            STYLE_WEIGHT * style_train_sparse
        ],
        format="csr",
        dtype=np.float32
    )

    X_combined_valid = hstack(
        [
            X_char_valid,
            STYLE_WEIGHT * style_valid_sparse
        ],
        format="csr",
        dtype=np.float32
    )

    print("Combined training shape:", X_combined_train.shape)
    print("Combined validation shape:", X_combined_valid.shape)

    # --------------------------------------------------------
    # TEST C VALUES
    # --------------------------------------------------------

    for c_value in C_VALUES:

        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000
        )

        model.fit(X_combined_train, y_train)

        predictions = model.predict(X_combined_valid)

        score = balanced_accuracy_score(
            y_valid,
            predictions
        )

        oof_predictions[c_value][valid_idx] = predictions

        fold_results.append({
            "fold": fold,
            "C": c_value,
            "balanced_accuracy": score
        })

        print(
            f"C = {c_value:<5} | "
            f"Balanced accuracy = {score:.5f}"
        )

    print(
        f"Fold execution time: "
        f"{time.time() - fold_start:.1f} seconds"
    )
    print()


# ============================================================
# SUMMARIZE RESULTS
# ============================================================

print("=" * 75)
print("FINAL C TUNING RESULTS")
print("=" * 75)

fold_results_df = pd.DataFrame(fold_results)

summary_rows = []

for c_value in C_VALUES:

    predictions = oof_predictions[c_value]

    oof_score = balanced_accuracy_score(
        y,
        predictions
    )

    fold_scores = fold_results_df.loc[
        fold_results_df["C"] == c_value,
        "balanced_accuracy"
    ].to_numpy()

    summary_rows.append({
        "C": c_value,
        "oof_balanced_accuracy": oof_score,
        "mean_fold_accuracy": fold_scores.mean(),
        "fold_std": fold_scores.std()
    })

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    "oof_balanced_accuracy",
    ascending=False
)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}"
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

fold_results_path = OUTPUT_DIR / "c_tuning_fold_results.csv"
summary_path = OUTPUT_DIR / "c_tuning_summary.csv"
oof_path = OUTPUT_DIR / "c_tuning_oof_predictions.csv"

fold_results_df.to_csv(
    fold_results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

oof_df = pd.DataFrame({
    "id": df["id"],
    "prompt_id": df["prompt_id"],
    "actual_model": y
})

for c_value in C_VALUES:
    oof_df[f"pred_C_{c_value}"] = oof_predictions[c_value]

oof_df.to_csv(
    oof_path,
    index=False
)


# ============================================================
# FINISH
# ============================================================

best_row = summary_df.iloc[0]

print()
print("=" * 75)
print("EXPERIMENT COMPLETE")
print("=" * 75)

print(f"Best C value: {best_row['C']}")

print(
    "Best OOF balanced accuracy: "
    f"{best_row['oof_balanced_accuracy']:.5f}"
)

print(f"Fold results saved: {fold_results_path}")
print(f"Summary saved: {summary_path}")
print(f"OOF predictions saved: {oof_path}")

print(
    f"Total execution time: "
    f"{(time.time() - start_all) / 60:.2f} minutes"
)

print()
print("All results saved successfully!")