"""
NAME THAT LLM - PREDICTION PIPELINE 2

Experiment 22 inspired:
- Multiple TF-IDF views
- LinearSVC base classifiers
- OOF stacking
- Logistic Regression meta-classifier
- Raw style features
- Independent Argmax predictions

Output:
outputs/submission_2.csv
"""

from pathlib import Path
import re
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSION_PATH = OUTPUT_DIR / "submission_2.csv"


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

INNER_SPLITS = 4

SVC_C = 0.5

META_C = 0.1

HEAD_TAIL_CHARS = 200

print("=" * 70)
print("NAME THAT LLM - PREDICTION PIPELINE 2")
print("Experiment 22 inspired stacking")
print("=" * 70)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n[1/7] Loading datasets...")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(TRAIN_PATH)

if not TEST_PATH.exists():
    raise FileNotFoundError(TEST_PATH)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

required_train = {"text", "model", "prompt_id"}
required_test = {"id", "text"}

if not required_train.issubset(train_df.columns):
    raise ValueError(
        f"Training dataset missing columns: "
        f"{required_train - set(train_df.columns)}"
    )

if not required_test.issubset(test_df.columns):
    raise ValueError(
        f"Test dataset missing columns: "
        f"{required_test - set(test_df.columns)}"
    )

train_df["text"] = train_df["text"].fillna("").astype(str)
test_df["text"] = test_df["text"].fillna("").astype(str)

if train_df["model"].isna().any():
    raise ValueError("Missing model labels in training data.")

if train_df["prompt_id"].isna().any():
    raise ValueError("Missing prompt IDs in training data.")

print("Training shape:", train_df.shape)
print("Testing shape :", test_df.shape)
print("Unique models :", train_df["model"].nunique())
print("Unique prompts:", train_df["prompt_id"].nunique())


# ============================================================
# 4. LABEL ENCODING
# ============================================================

print("\n[2/7] Encoding labels...")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    train_df["model"].values
)

classes = label_encoder.classes_

N_CLASSES = len(classes)

print("Number of classes:", N_CLASSES)


# ============================================================
# 5. TEXT TRANSFORMATIONS
# ============================================================

print("\n[3/7] Preparing text transformations...")


FUNCTION_WORDS = set("""
a an the and or but if while although because as so than that which who whom
whose this these those it its they them their theirs he him his she her hers
we us our ours you your yours i me my mine in on at to from with without for
of by over under between through into about is are was were be been being have
has had do does did can could will would shall should may might must not no nor
all any some each every both either neither other another such only also just
very too more most less least here there when where why how what whether then
now still yet even much many few little
""".split())


TOKEN_RE = re.compile(
    r"\w+|[^\w\s]",
    flags=re.UNICODE
)

LETTER_RE = re.compile(
    r"[^\W\d_]+",
    flags=re.UNICODE
)

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF]"
)


def shape_view(text):

    return LETTER_RE.sub(
        lambda m: (
            "X"
            if m.group(0)[0].isupper()
            else "x"
        ),
        text
    )


def function_view(text):

    output = []

    for token in TOKEN_RE.findall(text):

        if token.isalnum():

            if token.lower() in FUNCTION_WORDS:
                output.append(token.lower())
            else:
                output.append("CONTENT")

        else:
            output.append(token)

    return " ".join(output)


# ============================================================
# 6. STYLE FEATURES
# ============================================================

def style_features(t):

    s = t.rstrip()

    words = t.split()

    sentences = [
        x for x in re.split(r"[.!?]+\s", t)
        if x.strip()
    ]

    return {

        "log_chars": np.log1p(len(t)),

        "log_words": np.log1p(len(words)),

        "n_lines": t.count("\n"),

        "n_paragraphs": t.count("\n\n"),

        "n_bullets": len(
            re.findall(r"^\s*[-*\u2022]\s", t, re.M)
        ),

        "n_numbered": len(
            re.findall(r"^\s*\d+[.)]\s", t, re.M)
        ),

        "n_headers": len(
            re.findall(r"^#{1,6}\s", t, re.M)
        ),

        "n_bold": t.count("**"),

        "n_codefence": t.count("```"),

        "n_pipes": t.count("|"),

        "n_hr": len(
            re.findall(r"^\s*(---|\*\*\*)\s*$", t, re.M)
        ),

        "n_emdash": t.count("\u2014"),

        "n_endash": t.count("\u2013"),

        "n_curly": sum(
            t.count(c) for c in "\u2019\u201c\u201d"
        ),

        "n_straight_quote": (
            t.count('"') + t.count("'")
        ),

        "n_emoji": len(
            EMOJI_RE.findall(t)
        ),

        "n_latex": (
            t.count("$") + t.count("\\(")
        ),

        "n_colon": t.count(":"),

        "n_semicolon": t.count(";"),

        "n_excl": t.count("!"),

        "n_question": t.count("?"),

        "n_paren": t.count("("),

        "ends_punct": int(
            s[-1:] in ".!?)\"'`"
        ),

        "ends_question": int(
            s.endswith("?")
        ),

        "trailing_newline": int(
            t.endswith("\n")
        ),

        "leading_space": int(
            t[:1].isspace()
        ),

        "double_space": t.count("  "),

        "avg_word_len": (
            np.mean([len(w) for w in words])
            if words else 0
        ),

        "unique_ratio": (
            len(set(w.lower() for w in words))
            / max(len(words), 1)
        ),

        "avg_sent_words": (
            len(words) / max(len(sentences), 1)
        ),

        "upper_ratio": (
            sum(c.isupper() for c in t)
            / max(len(t), 1)
        ),

        "digit_ratio": (
            sum(c.isdigit() for c in t)
            / max(len(t), 1)
        ),

        "non_ascii_ratio": (
            sum(ord(c) > 127 for c in t)
            / max(len(t), 1)
        )

    }


print("Extracting raw style features...")

style_train = pd.DataFrame(
    [
        style_features(t)
        for t in train_df["text"]
    ]
).values.astype(np.float64)

style_test = pd.DataFrame(
    [
        style_features(t)
        for t in test_df["text"]
    ]
).values.astype(np.float64)

print("Training style shape:", style_train.shape)
print("Testing style shape :", style_test.shape)


# ============================================================
# 7. BUILD TF-IDF VIEWS
# ============================================================

print("\n[4/7] Building TF-IDF representations...")


VIEWS = [

    (
        "char",
        None,
        dict(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=250000
        )
    ),

    (
        "word",
        None,
        dict(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=180000
        )
    ),

    (
        "shape",
        shape_view,
        dict(
            analyzer="char",
            ngram_range=(2, 6),
            max_features=60000
        )
    ),

    (
        "function",
        function_view,
        dict(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=50000
        )
    )

]


CONCAT_WEIGHTS = {

    "char": 1.0,

    "word": 0.7,

    "shape": 0.5,

    "function": 0.5

}


train_texts = train_df["text"].tolist()

test_texts = test_df["text"].tolist()

views = {}

vectorizers = {}


for name, transform, config in VIEWS:

    print(f"Building {name} TF-IDF...")

    if transform is not None:

        tr = [
            transform(t)
            for t in train_texts
        ]

        te = [
            transform(t)
            for t in test_texts
        ]

    else:

        tr = train_texts

        te = test_texts

    kwargs = dict(

        lowercase=False,

        sublinear_tf=True,

        min_df=2,

        dtype=np.float32,

        **config

    )

    if config["analyzer"] == "word":

        kwargs["token_pattern"] = (
            r"(?u)\b\w+\b|[^\w\s]"
        )

    vectorizer = TfidfVectorizer(
        **kwargs
    )

    X_train = vectorizer.fit_transform(tr)

    X_test = vectorizer.transform(te)

    views[name] = (
        X_train,
        X_test
    )

    vectorizers[name] = vectorizer

    print(
        f"   {name}: "
        f"train={X_train.shape}, "
        f"test={X_test.shape}"
    )


# Concatenated view

print("\nBuilding concatenated TF-IDF...")

from scipy.sparse import hstack

X_concat_train = hstack(

    [
        views[name][0] * weight

        for name, weight in CONCAT_WEIGHTS.items()

    ],

    format="csr"

)

X_concat_test = hstack(

    [
        views[name][1] * weight

        for name, weight in CONCAT_WEIGHTS.items()

    ],

    format="csr"

)

views["concat"] = (
    X_concat_train,
    X_concat_test
)

BASE_MODELS = [
    "char",
    "word",
    "function",
    "concat"
]

print(
    "Concatenated shape:",
    X_concat_train.shape
)


# ============================================================
# 8. BASE MODEL SCORING
# ============================================================

def get_svc_scores(
    X_fit,
    y_fit,
    X_predict
):

    model = LinearSVC(

        C=SVC_C,

        class_weight="balanced",

        random_state=SEED,

        max_iter=10000

    )

    model.fit(
        X_fit,
        y_fit
    )

    scores = np.full(

        (
            X_predict.shape[0],
            N_CLASSES
        ),

        -2.0,

        dtype=np.float64

    )

    scores[:, model.classes_] = (
        model.decision_function(X_predict)
    )

    return scores


# ============================================================
# 9. GENERATE INNER OOF BASE SCORES
# ============================================================

print("\n[5/7] Generating OOF stacking features...")

groups = train_df["prompt_id"].to_numpy()

splitter = StratifiedGroupKFold(

    n_splits=INNER_SPLITS,

    shuffle=True,

    random_state=SEED

)

inner_splits = list(

    splitter.split(

        np.zeros(len(y)),

        y,

        groups

    )

)

meta_train_parts = []

meta_test_parts = []


for name in BASE_MODELS:

    print(
        f"\nTraining base learner: {name}"
    )

    X_train, X_test = views[name]

    oof_scores = np.zeros(

        (
            len(train_df),
            N_CLASSES
        ),

        dtype=np.float64

    )

    # Generate OOF predictions

    for fold, (tr_idx, va_idx) in enumerate(
        inner_splits,
        start=1
    ):

        print(
            f"   Inner fold {fold}/{INNER_SPLITS}"
        )

        scores = get_svc_scores(

            X_train[tr_idx],

            y[tr_idx],

            X_train[va_idx]

        )

        oof_scores[va_idx] = scores

    meta_train_parts.append(
        oof_scores
    )

    # Train full base model for test predictions

    print(
        f"   Training full {name} model..."
    )

    test_scores = get_svc_scores(

        X_train,

        y,

        X_test

    )

    meta_test_parts.append(
        test_scores
    )


# ============================================================
# 10. META-FEATURE CONSTRUCTION
# ============================================================

print("\n[6/7] Constructing meta-features...")

meta_train_parts.append(
    style_train
)

meta_test_parts.append(
    style_test
)

META_TRAIN = np.hstack(
    meta_train_parts
)

META_TEST = np.hstack(
    meta_test_parts
)

print(
    "Meta training shape:",
    META_TRAIN.shape
)

print(
    "Meta testing shape:",
    META_TEST.shape
)


# ============================================================
# 11. TRAIN META CLASSIFIER
# ============================================================

print("\nTraining Logistic Regression meta-classifier...")

scaler = StandardScaler()

META_TRAIN_SCALED = scaler.fit_transform(
    META_TRAIN
)

META_TEST_SCALED = scaler.transform(
    META_TEST
)

meta_model = LogisticRegression(

    C=META_C,

    class_weight="balanced",

    max_iter=3000,

    random_state=SEED

)

meta_model.fit(

    META_TRAIN_SCALED,

    y

)


# ============================================================
# 12. GENERATE FINAL PREDICTIONS
# ============================================================

print("\n[7/7] Generating final predictions...")

probabilities = np.full(

    (
        len(test_df),
        N_CLASSES
    ),

    1e-6,

    dtype=np.float64

)

probabilities[:, meta_model.classes_] = (
    meta_model.predict_proba(
        META_TEST_SCALED
    )
)

predictions = probabilities.argmax(
    axis=1
)

predicted_labels = label_encoder.inverse_transform(
    predictions
)


# ============================================================
# 13. CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({

    "id": test_df["id"],

    "model": predicted_labels

})

submission.to_csv(

    SUBMISSION_PATH,

    index=False

)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)

print("PREDICTION SUMMARY")

print("=" * 70)

print(
    "Test rows:",
    len(test_df)
)

print(
    "Predictions:",
    len(predicted_labels)
)

print("\nPredicted models:")

for sample_id, label in zip(
    test_df["id"],
    predicted_labels
):

    print(
        f"ID {sample_id}: {label}"
    )

print("\nSubmission preview:")

print(
    submission.to_string(index=False)
)

print("\nSaved submission to:")

print(SUBMISSION_PATH)

print("\n" + "=" * 70)

print("PREDICTION PIPELINE 2 COMPLETED")

print("=" * 70)