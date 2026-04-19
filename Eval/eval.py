import os
import json
import csv
import collections
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.corpus import stopwords
import re
import string
import sys
# ---------------------------
# Helper functions for metrics
# ---------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def preprocess_text(text):
    """Cleans and preprocesses text by removing special characters and stopwords, and applying stemming."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    words = text.split()  # Tokenize text
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords & apply stemming
    return " ".join(words)  # Return cleaned text

def compute_substring_match(gold, pred):
    """Computes a lenient substring match by considering word overlap in both directions with stemming."""
    gold, pred = preprocess_text(gold), preprocess_text(pred)
    gold_words, pred_words = set(gold.split()), set(pred.split())

    if not pred_words:
        return 0.0
    if not gold_words:
        return 0.0
    overlap_pred = sum(1 for word in pred_words if any(word in g for g in gold_words)) / len(pred_words)
    overlap_gold = sum(1 for word in gold_words if any(word in p for p in pred_words)) / len(gold_words)

    return max(overlap_pred, overlap_gold)  # Averaging both overlaps

def compute_exact_match(gold, pred):
    """Computes a lenient exact match score based on word overlap with stemming."""
    gold, pred = preprocess_text(gold), preprocess_text(pred)
    gold_words, pred_words = set(gold.split()), set(pred.split())
    return(1 if gold_words==pred_words else 0)
    #return sum(1 for word in pred_words if any(word in g for g in gold_words)) / len(pred_words) if pred_words else 0.0

def compute_f1(gold, pred):
    gold, pred = preprocess_text(gold), preprocess_text(pred)
    gold_tokens, pred_tokens = gold.lower().split(), pred.lower().split()
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    if(precision+recall==0):
        return 0
    return 2 * precision * recall / (precision + recall)

def compute_bleu(gold, pred):
    gold, pred = preprocess_text(gold), preprocess_text(pred)
    smoothie = SmoothingFunction().method4
    return sentence_bleu([gold.lower().split()], pred.lower().split(), smoothing_function=smoothie)

def compute_rouge_n(gold, pred, n):
    gold, pred = preprocess_text(gold), preprocess_text(pred)
    def get_ngrams(text, n):
        tokens = text.lower().split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    gold_ngrams, pred_ngrams = get_ngrams(gold, n), get_ngrams(pred, n)
    if len(gold_ngrams) == 0:
        return 0.0
    common = sum((collections.Counter(gold_ngrams) & collections.Counter(pred_ngrams)).values())
    return common / len(gold_ngrams)

def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[m][n]

def compute_rouge_l(gold, pred):
    gold, pred = preprocess_text(gold), preprocess_text(pred)
    gold_tokens, pred_tokens = gold.lower().split(), pred.lower().split()
    return lcs(gold_tokens, pred_tokens) / len(gold_tokens) if len(gold_tokens) > 0 else 0.0

def compute_metrics(gold, pred):
    return {
        'exact': 0.0 if not pred else compute_exact_match(gold, pred),
        'substring': 0.0 if not pred else compute_substring_match(gold, pred),
        'f1': 0.0 if not pred else compute_f1(gold, pred),
        'bleu': 0.0 if not pred else compute_bleu(gold, pred),
        'rouge1': 0.0 if not pred else compute_rouge_n(gold, pred, 1),
        'rouge2': 0.0 if not pred else compute_rouge_n(gold, pred, 2),
        'rougeL': 0.0 if not pred else compute_rouge_l(gold, pred)
    }

def average_metrics(metrics_list):
    if not metrics_list:
        return {key: 0.0 for key in ["exact", "substring", "f1", "bleu", "rouge1", "rouge2", "rougeL"]}
    return {key: sum(m[key] for m in metrics_list) / len(metrics_list) for key in metrics_list[0]}

def safe_load_json(file_path):
    """Safely loads JSON, handling missing files and malformed data."""
    if not os.path.exists(file_path):
        return None  # Indicate the file is missing
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return []

def main():
    gold_dir = r"./Questions"
    classified_dir = r"./Question-Metadata"
    evals_dir = r"./Output Folder"

    gold_files = [f for f in os.listdir(gold_dir) if f.endswith('.json')]
    eval_subfolders = [d for d in os.listdir(evals_dir) if os.path.isdir(os.path.join(evals_dir, d))]

    eval_models_atype = {subfolder: collections.defaultdict(list) for subfolder in eval_subfolders}
    answer_type_pattern = re.compile(r'^Question \d+ Reasoning Type$')

    for file in gold_files:
        gold_data = safe_load_json(os.path.join(gold_dir, file))
        classified_data = safe_load_json(os.path.join(classified_dir, file))

        if not gold_data or not classified_data:
            continue  

        for subfolder in eval_subfolders:
            eval_path = os.path.join(evals_dir, subfolder, file)
            eval_data = safe_load_json(eval_path)

            if eval_data is None:
                continue  

            if eval_data == []:  
                num_questions = min(len(gold_data), len(classified_data))
                for i in range(num_questions):
                    a_type = next((value for key, value in classified_data[i].items() if answer_type_pattern.match(key)), "Unknown")
                    eval_models_atype[subfolder][a_type].append({'exact': 0.0, 'substring': 0.0, 'f1': 0.0, 'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0})
                continue

            num_answers = min(len(gold_data), len(classified_data), len(eval_data))

            for i in range(num_answers):
                try:
                    gold_answer = str(list(gold_data[i].values())[1]) if isinstance(gold_data[i], dict) and len(gold_data[i]) > 1 else str(gold_data[i])
                    eval_answer = str(list(eval_data[i].values())[1]) if isinstance(eval_data[i], dict) and len(eval_data[i]) > 1 else str(eval_data[i]) if isinstance(eval_data[i], str) else ""

                    a_type = next((value for key, value in classified_data[i].items() if answer_type_pattern.match(key)), "Unknown")
                    if a_type == "Unknown":
                        continue

                    metrics = compute_metrics(gold_answer, eval_answer)
                    eval_models_atype[subfolder][a_type].append(metrics)

                except IndexError:
                    print(f"⚠️ Skipping index {i} in {subfolder}  {file} due to missing data.")
                except Exception as e:
                    print(f"⚠️ Unexpected error processing {subfolder}  {file}: {e}")

    def save_to_csv(filename, data):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            metrics = ["exact", "substring", "f1"]

            categories = sorted({key for model in data.values() for key in model.keys()})

            writer.writerow(["Model"] + [cat for cat in categories for _ in metrics])
            writer.writerow([""] + metrics * len(categories))

            for model, categories_data in data.items():
                row = [model]
                for cat in categories:
                    avg_metrics = average_metrics(categories_data.get(cat, []))
                    row.extend([f"{avg_metrics[key]:.3f}" for key in metrics])
                writer.writerow(row)
    save_to_csv("RT-1-Combined-Human.csv", eval_models_atype)
    print("\n✅ Results saved to QT-Human.csv!")

if __name__ == "__main__":
    main()