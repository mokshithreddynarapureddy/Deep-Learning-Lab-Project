from pathlib import Path
import re
import string
import warnings
import itertools

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
N_SPLITS = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("NAME THAT LLM - ACCURACY OPTIMIZATION")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)

train["text"] = train["text"].fillna("").astype(str)
train["model"] = train["model"].astype(str)

X = train["text"].tolist()
y = train["model"].to_numpy()
groups = train["prompt_id"].to_numpy()

print("\nTraining rows:", len(train))
print("Classes:", len(np.unique(y)))
print("Unique prompts:", len(np.unique(groups)))


# ============================================================
# ORIGINAL 28 STYLE FEATURES
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

    inline_code_count = (
        text.count("`") - code_fence_count * 3
    )

    url_count = len(
        re.findall(r"https?://\S+|www\.\S+", text)
    )

    return [
        chars,
        word_count,
        sentence_count,
        paragraph_count,
        newline_count,
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
        exclamation_count / max(word_count, 1),
        question_count / max(word_count, 1),
        comma_count / max(word_count, 1),
        semicolon_count / max(word_count, 1),
        colon_count / max(word_count, 1),
        quote_count / max(word_count, 1),
        markdown_heading_count,
        bullet_count,
        numbered_list_count,
        code_fence_count,
        inline_code_count,
        url_count,
    ]


# ============================================================
# ADDITIONAL FEATURE VOCABULARIES
# ============================================================

FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "if", "because",
    "although", "while", "when", "where", "which", "that",
    "who", "what", "how", "why", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "can", "could", "will", "would", "should", "may",
    "might", "must", "not", "no", "yes", "to", "of", "in",
    "on", "at", "by", "for", "from", "with", "without",
    "as", "into", "about", "over", "under", "between",
    "through", "this", "these", "those", "it", "its", "they",
    "them", "their", "we", "us", "our", "you", "your", "i",
    "me", "my", "he", "she", "his", "her", "also", "just",
    "only", "very", "more", "most", "some", "any", "all",
    "each", "every", "such", "there", "here", "then", "than",
]

DISCOURSE_MARKERS = [
    "however", "therefore", "moreover", "furthermore",
    "nevertheless", "consequently", "instead", "otherwise",
    "meanwhile", "additionally", "finally", "firstly",
    "secondly", "ultimately", "indeed", "notably",
    "specifically", "essentially", "generally", "overall",
    "in conclusion", "for example", "for instance",
    "in fact", "on the other hand", "as a result",
]

HEDGING_WORDS = [
    "perhaps", "possibly", "probably", "might", "may",
    "could", "likely", "unlikely", "seems", "appears",
    "generally", "typically", "often", "sometimes",
    "arguably", "approximately", "roughly", "potentially",
]

CONTRACTIONS = [
    "don't", "doesn't", "didn't", "can't", "couldn't",
    "won't", "wouldn't", "shouldn't", "isn't", "aren't",
    "wasn't", "weren't", "it's", "that's", "there's",
    "they're", "we're", "you're", "i'm", "i've", "we've",
    "they've", "you've", "i'll", "we'll", "they'll",
    "you'll", "i'd", "we'd", "they'd", "you'd",
]


# ============================================================
# ADDITIONAL 31 STRUCTURAL FEATURES
# ============================================================

def extract_new_features(text):

    text = str(text)
    lower = text.lower()

    tokens = re.findall(
        r"\b[\w]+(?:['’][\w]+)?\b",
        lower,
    )

    word_count = len(tokens)
    unique_tokens = set(tokens)

    token_counts = {}

    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]

    sentence_lengths = [
        len(re.findall(r"\b\w+\b", sentence))
        for sentence in sentences
    ]

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    paragraph_lengths = [
        len(re.findall(r"\b\w+\b", paragraph))
        for paragraph in paragraphs
    ]

    repeated_tokens = sum(
        count > 1 for count in token_counts.values()
    )

    hapax_tokens = sum(
        count == 1 for count in token_counts.values()
    )

    function_word_count = sum(
        token_counts.get(word, 0)
        for word in FUNCTION_WORDS
    )

    discourse_count = sum(
        lower.count(phrase)
        for phrase in DISCOURSE_MARKERS
    )

    hedging_count = sum(
        token_counts.get(word, 0)
        for word in HEDGING_WORDS
    )

    contraction_count = sum(
        lower.count(word)
        for word in CONTRACTIONS
    )

    pronouns = [
        "i", "me", "my", "mine", "we", "us", "our",
        "you", "your", "he", "him", "his", "she",
        "her", "they", "them", "their", "it",
    ]

    pronoun_count = sum(
        token_counts.get(word, 0)
        for word in pronouns
    )

    ellipsis_count = text.count("...") + text.count("…")

    dash_count = (
        text.count("-")
        + text.count("—")
        + text.count("–")
    )

    parentheses_count = text.count("(") + text.count(")")
    bracket_count = text.count("[") + text.count("]")
    brace_count = text.count("{") + text.count("}")
    slash_count = text.count("/")
    asterisk_count = text.count("*")

    markdown_link_count = len(
        re.findall(r"\[[^\]]+\]\([^)]+\)", text)
    )

    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    spaces = sum(c.isspace() for c in text)

    denominator = max(word_count, 1)
    char_denominator = max(len(text), 1)

    return [
        len(unique_tokens) / denominator,
        repeated_tokens / max(len(unique_tokens), 1),
        hapax_tokens / denominator,

        function_word_count / denominator,
        discourse_count / denominator,
        hedging_count / denominator,
        contraction_count / denominator,
        pronoun_count / denominator,

        np.mean(sentence_lengths) if sentence_lengths else 0,
        np.std(sentence_lengths) if sentence_lengths else 0,
        np.median(sentence_lengths) if sentence_lengths else 0,
        np.percentile(sentence_lengths, 90) if sentence_lengths else 0,
        max(sentence_lengths) if sentence_lengths else 0,

        np.mean(paragraph_lengths) if paragraph_lengths else 0,
        np.std(paragraph_lengths) if paragraph_lengths else 0,
        np.median(paragraph_lengths) if paragraph_lengths else 0,
        max(paragraph_lengths) if paragraph_lengths else 0,

        ellipsis_count / denominator,
        dash_count / denominator,
        parentheses_count / denominator,
        bracket_count / denominator,
        brace_count / denominator,
        slash_count / denominator,
        asterisk_count / denominator,
        markdown_link_count / denominator,

        letters / char_denominator,
        digits / char_denominator,
        spaces / char_denominator,

        len(re.findall(r"(?m)^\s*$", text)),
        len(re.findall(r"(?m)^>+\s?", text)),
        len(re.findall(r"(?m)^\s*[-*+]\s+\[[ xX]\]", text)),
    ]


# ============================================================
# EXTRACT NUMERICAL FEATURES
# ============================================================

print("\nExtracting numerical features...")

style_raw = np.asarray(
    [extract_style_features(t) for t in X],
    dtype=np.float32,
)

new_raw = np.asarray(
    [extract_new_features(t) for t in X],
    dtype=np.float32,
)

style_raw = np.nan_to_num(
    style_raw, nan=0, posinf=0, neginf=0
)

new_raw = np.nan_to_num(
    new_raw, nan=0, posinf=0, neginf=0
)

print("Style features:", style_raw.shape)
print("Structural features:", new_raw.shape)


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

CONFIGS = [
    # char range, word range, word weight,
    # style weight, structural weight, C

    ((2, 5), (1, 2), 0.10, 0.50, 0.25, 1.0),
    ((2, 5), (1, 2), 0.25, 0.50, 0.25, 1.0),
    ((2, 5), (1, 2), 0.50, 0.50, 0.25, 1.0),

    ((2, 6), (1, 2), 0.25, 0.50, 0.25, 1.0),
    ((3, 6), (1, 2), 0.25, 0.50, 0.25, 1.0),

    ((2, 5), (1, 3), 0.25, 0.50, 0.25, 1.0),

    ((2, 5), (1, 2), 0.25, 0.25, 0.25, 1.0),
    ((2, 5), (1, 2), 0.25, 0.75, 0.25, 1.0),

    ((2, 5), (1, 2), 0.25, 0.50, 0.10, 1.0),
    ((2, 5), (1, 2), 0.25, 0.50, 0.40, 1.0),

    ((2, 5), (1, 2), 0.25, 0.50, 0.25, 0.5),
    ((2, 5), (1, 2), 0.25, 0.50, 0.25, 2.0),

    ((2, 5), (1, 2), 0.25, 0.50, 0.25, 3.0),
]


# ============================================================
# GROUP K-FOLD VALIDATION
# ============================================================

cv = GroupKFold(n_splits=N_SPLITS)

splits = list(cv.split(X, y, groups))

results = []

best_score = -1
best_config = None
best_oof = None

print("\nTotal configurations:", len(CONFIGS))
print("Total folds:", N_SPLITS)

print("\nStarting optimization...\n")


# ============================================================
# CACHE TF-IDF MATRICES FOR EACH FOLD
# ============================================================

for fold_number, (train_idx, val_idx) in enumerate(splits, 1):

    print("\n" + "=" * 70)
    print(f"PREPARING FOLD {fold_number}")
    print("=" * 70)

    X_fold_train = [X[i] for i in train_idx]
    X_fold_val = [X[i] for i in val_idx]

    y_fold_train = y[train_idx]
    y_fold_val = y[val_idx]

    # Fit scalers only on fold training data.

    style_scaler = StandardScaler()

    style_train = style_scaler.fit_transform(
        style_raw[train_idx]
    )

    style_val = style_scaler.transform(
        style_raw[val_idx]
    )

    style_train = normalize(
        csr_matrix(style_train, dtype=np.float32)
    )

    style_val = normalize(
        csr_matrix(style_val, dtype=np.float32)
    )

    new_scaler = StandardScaler()

    new_train = new_scaler.fit_transform(
        new_raw[train_idx]
    )

    new_val = new_scaler.transform(
        new_raw[val_idx]
    )

    new_train = normalize(
        csr_matrix(new_train, dtype=np.float32)
    )

    new_val = normalize(
        csr_matrix(new_val, dtype=np.float32)
    )

    # --------------------------------------------------------
    # Cache character vectorizers
    # --------------------------------------------------------

    char_cache = {}

    char_ranges = sorted(
        set(config[0] for config in CONFIGS)
    )

    for ngram_range in char_ranges:

        print("Building character TF-IDF:", ngram_range)

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=2,
            max_features=250_000,
            sublinear_tf=True,
            max_df=0.98,
            dtype=np.float32,
        )

        char_cache[ngram_range] = (
            vectorizer.fit_transform(X_fold_train),
            vectorizer.transform(X_fold_val),
        )

    # --------------------------------------------------------
    # Cache word vectorizers
    # --------------------------------------------------------

    word_cache = {}

    word_ranges = sorted(
        set(config[1] for config in CONFIGS)
    )

    for ngram_range in word_ranges:

        print("Building word TF-IDF:", ngram_range)

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            min_df=2,
            max_features=150_000,
            sublinear_tf=True,
            max_df=0.98,
            dtype=np.float32,
            strip_accents=None,
        )

        word_cache[ngram_range] = (
            vectorizer.fit_transform(X_fold_train),
            vectorizer.transform(X_fold_val),
        )

    # --------------------------------------------------------
    # Evaluate configurations
    # --------------------------------------------------------

    for config_number, config in enumerate(CONFIGS, 1):

        (
            char_range,
            word_range,
            word_weight,
            style_weight,
            new_weight,
            C,
        ) = config

        char_train, char_val = char_cache[char_range]

        word_train, word_val = word_cache[word_range]

        combined_train = hstack(
            [
                char_train,
                word_train * word_weight,
                style_train * style_weight,
                new_train * new_weight,
            ],
            format="csr",
        )

        combined_val = hstack(
            [
                char_val,
                word_val * word_weight,
                style_val * style_weight,
                new_val * new_weight,
            ],
            format="csr",
        )

        model = LinearSVC(
            C=C,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5000,
        )

        model.fit(combined_train, y_fold_train)

        predictions = model.predict(combined_val)

        accuracy = accuracy_score(
            y_fold_val, predictions
        )

        balanced = balanced_accuracy_score(
            y_fold_val, predictions
        )

        results.append({
            "fold": fold_number,
            "config": config_number,
            "char_range": str(char_range),
            "word_range": str(word_range),
            "word_weight": word_weight,
            "style_weight": style_weight,
            "new_weight": new_weight,
            "C": C,
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
        })

        print(
            f"Config {config_number:02d}/{len(CONFIGS)} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Balanced: {balanced:.4f}"
        )


# ============================================================
# SUMMARIZE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby("config")
    .agg(
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
    )
    .reset_index()
)

summary = summary.merge(
    results_df.drop_duplicates("config")[
        [
            "config",
            "char_range",
            "word_range",
            "word_weight",
            "style_weight",
            "new_weight",
            "C",
        ]
    ],
    on="config",
)

summary = summary.sort_values(
    "mean_balanced_accuracy",
    ascending=False,
)

results_df.to_csv(
    OUTPUT_DIR / "optimization_fold_results.csv",
    index=False,
)

summary.to_csv(
    OUTPUT_DIR / "optimization_summary.csv",
    index=False,
)


# ============================================================
# DISPLAY BEST CONFIGURATIONS
# ============================================================

print("\n" + "=" * 75)
print("OPTIMIZATION RESULTS")
print("=" * 75)

print(
    summary[
        [
            "config",
            "mean_accuracy",
            "mean_balanced_accuracy",
            "std_accuracy",
            "char_range",
            "word_range",
            "word_weight",
            "style_weight",
            "new_weight",
            "C",
        ]
    ].to_string(index=False)
)

best = summary.iloc[0]

print("\n" + "=" * 75)
print("BEST CONFIGURATION")
print("=" * 75)

print(best.to_string())

print("\nResults saved in:", OUTPUT_DIR)

print("\nIMPORTANT:")
print("These are cross-validation results.")
print("The final Kaggle score may differ.")