import re
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import random

RANDOM_STATE = 42

# --- 0. Paths: ---
TRAIN_PATH = "C:\\Users\\ramti\\OneDrive\\Desktop\\train.csv"   
VALID_PATH = "C:\\Users\\ramti\\OneDrive\\Desktop\\valid.csv"
TEST_PATH  = "C:\\Users\\ramti\\OneDrive\\Desktop\\test.csv"

VOCAB_PATH = "C:\\Users\\ramti\\OneDrive\\Desktop\\vocab.txt"
TRAIN_IDS_PATH = "C:\\Users\\ramti\\OneDrive\\Desktop\\train_ids.txt"
VALID_IDS_PATH = "C:\\Users\\ramti\\OneDrive\\Desktop\\valid_ids.txt"
TEST_IDS_PATH  = "C:\\Users\\ramti\\OneDrive\\Desktop\\test_ids.txt"

# --- 1. Preprocessing ---
def preprocess_text(text):
    # remove punctuation (keep alphanumerics and whitespace), lowercase
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # remove punctuation (keep underscores as word chars if present)
    text = re.sub(r'[^\w\s]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 2. Load data (expects columns 'text' and 'label') ---
def load_and_clean(path):
    df = pd.read_csv(path)  # adapt read_csv args if TSV or no header
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must have 'text' and 'label' columns.")
    df['text_clean'] = df['text'].astype(str).map(preprocess_text)
    return df[['text_clean', 'label']]

train_df = load_and_clean(TRAIN_PATH)
valid_df = load_and_clean(VALID_PATH)
test_df  = load_and_clean(TEST_PATH)

# Optional: inspect label range and convert if necessary
def normalize_label_scheme(df):
    # ensure labels are ints
    df['label'] = df['label'].astype(int)
    # If labels are 1..4 but you prefer 0..3 for sklearn, subtract 1:
    if df['label'].min() == 1 and df['label'].max() == 4:
        df['label'] = df['label'] - 1
    return df

train_df = normalize_label_scheme(train_df)
valid_df = normalize_label_scheme(valid_df)
test_df  = normalize_label_scheme(test_df)

# --- 3. Build vocabulary using training set only ---
def build_vocab(texts, max_features=10000):
    counter = Counter()
    for t in texts:
        tokens = t.split()
        counter.update(tokens)
    # take top max_features tokens by frequency
    most_common = counter.most_common(max_features)
    # assign IDs starting from 1 (as requested)
    word_to_id = {w: idx+1 for idx, (w, _) in enumerate(most_common)}
    # frequency dict
    freq = {w: c for w, c in most_common}
    return word_to_id, freq

word_to_id, word_freq = build_vocab(train_df['text_clean'].tolist(), max_features=10000)
vocab_size = len(word_to_id)
print(f"Vocabulary size: {vocab_size}")

# Save vocab file: "word \t id \t frequency"
with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
    for w, idx in word_to_id.items():
        f.write(f"{w}\t{idx}\t{word_freq[w]}\n")

# --- 4. Export ID-format data files ---
def text_to_id_list(text, word_to_id):
    tokens = text.split()
    ids = [str(word_to_id[t]) for t in tokens if t in word_to_id]
    return ids

def save_ids_file(df, out_path, word_to_id):
    with open(out_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            ids = text_to_id_list(row['text_clean'], word_to_id)
            label = int(row['label'])
            line = " ".join(ids) + "\t" + str(label) + "\n"
            f.write(line)

save_ids_file(train_df, TRAIN_IDS_PATH, word_to_id)
save_ids_file(valid_df, VALID_IDS_PATH, word_to_id)
save_ids_file(test_df,  TEST_IDS_PATH,  word_to_id)

# --- 5. Build sparse matrices (BBoW and FBoW) ---
def build_sparse_matrix(docs_texts, word_to_id, representation='binary'):
    """
    representation: 'binary' or 'freq'
    returns csr_matrix (n_docs, vocab_size)
    For 'freq' we normalize per document so row sums to 1 (over words present in vocab).
    """
    rows = []
    cols = []
    data = []
    n_docs = len(docs_texts)
    for i, text in enumerate(docs_texts):
        tokens = text.split()
        ids = [word_to_id[t] for t in tokens if t in word_to_id]
        if len(ids) == 0:
            continue
        if representation == 'binary':
            uniq_ids = set(ids)
            for wid in uniq_ids:
                rows.append(i)
                cols.append(wid-1)   # 0-based for matrix columns
                data.append(1.0)
        elif representation == 'freq':
            c = Counter(ids)  # counts for vocab words only
            total = sum(c.values())
            if total == 0:
                continue
            for wid, cnt in c.items():
                rows.append(i)
                cols.append(wid-1)
                data.append(cnt / total)  # so row sums to 1 (over vocab words in doc)
        else:
            raise ValueError("representation must be 'binary' or 'freq'")
    mat = csr_matrix((data, (rows, cols)), shape=(n_docs, len(word_to_id)), dtype=float)
    return mat

X_train_bin = build_sparse_matrix(train_df['text_clean'].tolist(), word_to_id, representation='binary')
X_valid_bin = build_sparse_matrix(valid_df['text_clean'].tolist(), word_to_id, representation='binary')
X_test_bin  = build_sparse_matrix(test_df['text_clean'].tolist(),  word_to_id, representation='binary')

X_train_freq = build_sparse_matrix(train_df['text_clean'].tolist(), word_to_id, representation='freq')
X_valid_freq = build_sparse_matrix(valid_df['text_clean'].tolist(), word_to_id, representation='freq')
X_test_freq  = build_sparse_matrix(test_df['text_clean'].tolist(),  word_to_id, representation='freq')

y_train = train_df['label'].values
y_valid = valid_df['label'].values
y_test  = test_df['label'].values

# --- 6. Train & evaluate helper ---
def evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, verbose=True):
    out = {}
    # predictions
    preds_train = model.predict(X_train)
    preds_valid = model.predict(X_valid)
    preds_test  = model.predict(X_test)
    out['f1_train'] = f1_score(y_train, preds_train, average='macro')
    out['f1_valid'] = f1_score(y_valid, preds_valid, average='macro')
    out['f1_test']  = f1_score(y_test, preds_test, average='macro')
    if verbose:
        print("F1 (macro) — train: {:.4f}, valid: {:.4f}, test: {:.4f}".format(
            out['f1_train'], out['f1_valid'], out['f1_test']))
        print("Classification report on test set:")
        print(classification_report(y_test, preds_test))
    return out

# --- 7. Models: simple baseline training + light grid search example ---

results = {}

# 7.a Logistic Regression baseline
lr = LogisticRegression(solver='saga', penalty='l2', C=1.0, max_iter=2000, random_state=RANDOM_STATE)
lr.fit(X_train_bin, y_train)   # try binary first
print("LogisticRegression (BBoW) baseline")
results['lr_bin_baseline'] = evaluate_model(lr, X_train_bin, y_train, X_valid_bin, y_valid, X_test_bin, y_test)



# 7.b Decision Tree
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt.fit(X_train_bin, y_train)
print("DecisionTree (BBoW) baseline")
results['dt_bin_baseline'] = evaluate_model(dt, X_train_bin, y_train, X_valid_bin, y_valid, X_test_bin, y_test)


# 7.c Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_bin, y_train)
print("RandomForest (BBoW) baseline")
results['rf_bin_baseline'] = evaluate_model(rf, X_train_bin, y_train, X_valid_bin, y_valid, X_test_bin, y_test)


# 7.d XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1)
xgb_clf.fit(X_train_bin, y_train)
print("XGBoost (BBoW) baseline")
results['xgb_bin_baseline'] = evaluate_model(xgb_clf, X_train_bin, y_train, X_valid_bin, y_valid, X_test_bin, y_test)





print("Done. Results keys:", list(results.keys()))
