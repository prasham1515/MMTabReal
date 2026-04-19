import os
import google.generativeai as genai
import pandas as pd
import json
import time
import argparse
from dotenv import load_dotenv

# GitHub-ready paths (resolved from repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "MMTabReal", "all")
QUESTIONS_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
OUTPUT_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Results", "lower-gemini")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "missing-image.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Ensure the output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Missing-image Gemini baseline runner")
    parser.add_argument(
        "--question-mode",
        default="all",
        choices=["all", "one-by-one"],
        help="Ask all questions in one request or ask one question per request",
    )
    return parser.parse_args()

def load_prompt_text(prompt_path, format_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    with open(format_path, "r", encoding="utf-8") as f:
        formatting = f.read().strip()
    return prompt_text.replace("{ANSWER_FORMATTING_GUIDELINES}", formatting)

def read_questions(questions_file):
    """Extract only the questions from a JSON file."""
    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [list(q.values())[0] for q in data]  # Extract only question text


def is_image_reference(value):
    if not isinstance(value, str):
        return False
    return value.strip().lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))


def csv_to_pipe_string(csv_path):
    """Convert CSV text data to a pipe-separated string, skipping image filename cells."""
    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip",
            skip_blank_lines=True,
            dtype=str,
        )
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    # Lower baseline: skip image entries entirely by blanking image filenames.
    df = df.applymap(lambda x: "" if is_image_reference(x) else x)
    df = df.applymap(lambda x: ''.join(char for char in str(x) if ord(char) < 128) if isinstance(x, str) else x)

    return df.to_csv(sep="|", index=False)  # Convert to pipe-separated format

def classify_with_gemini(base_prompt, data, questions, question_mode="all"):
    """Send the formatted data along with extracted questions to Gemini API."""
    model = genai.GenerativeModel("gemini-2.0-flash")

    if question_mode == "all":
        prompt = (
            base_prompt
            + f"\n\nHere is the dataset in a pipe-separated format:\n{data}\n\n"
            + "Answer the following questions based on the dataset:\n"
            + json.dumps(questions)
        )
        response = model.generate_content(prompt)
        return response.text
    else:
        results = []
        for idx, question in enumerate(questions, start=1):
            prompt = (
                base_prompt
                + f"\n\nHere is the dataset in a pipe-separated format:\n{data}\n\n"
                + "Answer the following question based on the dataset:\n"
                + question
            )
            response = model.generate_content(prompt)
            results.append({f"Question {idx}": question, f"Answer {idx}": response.text})
        return json.dumps(results)

def process_csv_files(question_mode="all"):
    """Process CSV files, extract questions, and send to Gemini."""
    base_prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)
    counter = 1

    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        csv_path = os.path.join(folder_path, f"{folder_name}.csv")
        output_file_path = os.path.join(OUTPUT_DIR, f"{folder_name}.json")

        if os.path.exists(output_file_path):
            print(f"Skipping {folder_name}, result already exists.")
            continue

        pipe_data = csv_to_pipe_string(csv_path)
        if pipe_data is None:
            continue

        questions_file = os.path.join(QUESTIONS_DIR, f"{folder_name}.json")
        if not os.path.exists(questions_file):
            print(f"No question file found for {folder_name}, skipping.")
            continue

        questions = read_questions(questions_file)

        # Call Gemini API with the processed data and questions
        gemini_response = classify_with_gemini(base_prompt, pipe_data, questions, question_mode)
        counter += 1

        # Clean and parse the response
        gemini_response = gemini_response.replace("```", "").replace("json", "")

        try:
            gemini_response = json.loads(gemini_response)
            # Save response as JSON
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(gemini_response, f, indent=4)
            print("Processed ", output_file_path)
        except json.JSONDecodeError:
            print(f"Error decoding response for {folder_name}")

if __name__ == "__main__":
    args = parse_args()
    process_csv_files(args.question_mode)
