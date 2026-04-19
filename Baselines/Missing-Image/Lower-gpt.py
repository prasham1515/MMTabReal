import os
import time
import json
import pandas as pd
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# GitHub-ready paths (resolved from repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "MMTabReal", "all")
QUESTIONS_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
OUTPUT_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Results", "lower-gpt")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "missing-image.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY                  = os.getenv("OPENAI_API_KEY")
MODEL                    = "gpt-4o-mini"  # or "gpt-4o"
RATE_LIMIT_CALLS         = 1000     # after this many calls...
RATE_LIMIT_PAUSE_SEC     = 60     # ...sleep this many seconds
# ── END CONFIG ────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Missing-image GPT baseline runner")
    parser.add_argument(
        "--question-mode",
        default="all",
        choices=["all", "one-by-one"],
        help="Ask all questions in one request or ask one question per request",
    )
    return parser.parse_args()

def read_questions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [list(item.values())[0] for item in data]


def load_prompt_text(prompt_path, format_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    with open(format_path, "r", encoding="utf-8") as f:
        formatting = f.read().strip()
    return prompt_text.replace("{ANSWER_FORMATTING_GUIDELINES}", formatting)


def is_image_reference(value):
    if not isinstance(value, str):
        return False
    return value.strip().lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))


def csv_to_pipe_string(csv_path):
    try:
        df = pd.read_csv(
            csv_path,
            engine='python',
            on_bad_lines='skip',
            skip_blank_lines=True,
            dtype=str
        )
    except Exception as e:
        print(f"[!] Error reading {csv_path}: {e}")
        return None

    # Lower baseline: skip image entries entirely by blanking image filenames.
    df = df.applymap(lambda x: "" if is_image_reference(x) else x)
    # strip non-ASCII
    df = df.applymap(lambda x: ''.join(ch for ch in str(x) if ord(ch) < 128) if isinstance(x, str) else x)
    return df.to_csv(sep="|", index=False)


def make_prompt(base_prompt, table_psv, questions):
    return (
        base_prompt
        + "\n\nHere is the dataset in a pipe-separated format:\n"
        + table_psv
        + "\n\nAnswer the following questions based on the dataset:\n"
        + json.dumps(questions)
    )

def main(question_mode="all"):
    client = OpenAI(api_key=API_KEY)
    base_prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)
    call_count = 0

    for folder in sorted(os.listdir(DATA_DIR)):
        subdir = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(subdir):
            continue

        csv_path = os.path.join(subdir, f"{folder}.csv")
        q_path   = os.path.join(QUESTIONS_DIR, f"{folder}.json")
        out_path = os.path.join(OUTPUT_DIR, f"{folder}.json")

        if not os.path.isfile(csv_path):
            print(f"[-] Skipping {folder}: no CSV found")
            continue
        if not os.path.isfile(q_path):
            print(f"[-] Skipping {folder}: no questions JSON")
            continue
        if os.path.exists(out_path):
            print(f"[=] Already done {folder}, skipping")
            continue

        table_psv = csv_to_pipe_string(csv_path)
        if table_psv is None:
            continue

        questions = read_questions(q_path)
        prompt    = make_prompt(base_prompt, table_psv, questions)

        if question_mode == "all":
            print(f"[→] Asking GPT for {folder} …")
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                messages=[{"role":"user","content":prompt}]
            )
            answer_text = resp.choices[0].message.content.strip()

            # clean and parse JSON
            answer_json = answer_text.replace("```json","").replace("```","")
            try:
                parsed = json.loads(answer_json)
            except json.JSONDecodeError as e:
                print(f"[!] JSON parse error for {folder}: {e}")
                parsed = {"error": "JSON parse failed", "raw": answer_text}
        else:
            parsed = []
            for idx, question in enumerate(questions, start=1):
                single_q_prompt = (
                    base_prompt
                    + "\n\nHere is the dataset in a pipe-separated format:\n"
                    + table_psv
                    + "\n\nAnswer the following question based on the dataset:\n"
                    + question
                )
                print(f"[→] Q{idx}/{len(questions)} for {folder} …")
                resp = client.chat.completions.create(
                    model=MODEL,
                    temperature=0.0,
                    messages=[{"role":"user","content":single_q_prompt}]
                )
                answer_text = resp.choices[0].message.content.strip()
                parsed.append({f"Question {idx}": question, f"Answer {idx}": answer_text})
            # save raw for inspection
            parsed = answer_text

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print(f"[✔] Saved answer to {out_path}")

        call_count += 1

if __name__ == "__main__":
    args = parse_args()
    main(args.question_mode)
        if call_count % RATE_LIMIT_CALLS == 0:
            print(f"[⏱] Reached {RATE_LIMIT_CALLS} calls, sleeping {RATE_LIMIT_PAUSE_SEC}s …")
            time.sleep(RATE_LIMIT_PAUSE_SEC)

    print("All done!")

if __name__ == "__main__":
    main()
