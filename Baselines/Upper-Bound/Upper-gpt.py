import os
from openai import OpenAI
import pandas as pd
import json
import time
import argparse
import sys
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key = API_KEY,
)

# GitHub-ready paths (resolved from repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
Upper_folder = os.path.join(ROOT_DIR, "MMTabReal", "Upper_Bound")
questions_folder = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
answer_path = os.path.join(ROOT_DIR, "MMTabReal", "Results", "upper-gpt")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "upper.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")

# Ensure the output folder exists
os.makedirs(answer_path, exist_ok=True)


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

def xlsx_to_pipe_string(xlsx_path):
    """Convert XLSX data to a pipe-separated string, using the full workbook contents."""
    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        print(f"Error reading {xlsx_path}: {e}")
        return None

    parts = []
    for sheet_name, df in sheets.items():
        df = df.map(lambda x: ''.join(char for char in str(x) if ord(char) < 128) if isinstance(x, str) else x)
        parts.append(f"[Sheet: {sheet_name}]\n{df.to_csv(sep='|', index=False)}")

    return "\n".join(parts)



def classify_with_gpt(base_prompt, data, questions, question_mode="all"):
    """Send formatted data and questions to GPT API."""
    
    if question_mode == "all":
        prompt = (
            base_prompt
            + f"\n\nHere is the dataset in a pipe-separated format:\n{data}\n\n"
            + "Answer the following questions based on the dataset:\n"
            + json.dumps(questions)
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    else:
        results = []
        for idx, question in enumerate(questions, start=1):
            prompt = (
                base_prompt
                + f"\n\nHere is the dataset in a pipe-separated format:\n{data}\n\n"
                + "Answer the following question based on the dataset:\n"
                + question
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            answer_text = response.choices[0].message.content
            results.append({f"Question {idx}": question, f"Answer {idx}": answer_text})
        
        return json.dumps(results)

def process_xlsx_files(question_mode="all"):
    """Process XLSX files, extract questions, and send to Gemini."""
    base_prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)
    counter = 1

    for filename in os.listdir(Upper_folder):
        if filename.endswith(".xlsx"):
            csv_path = os.path.join(Upper_folder, filename)
            output_file_path = os.path.join(answer_path, filename.replace(".xlsx", ".json"))

            # Convert XLSX to pipe-separated string
            pipe_data = xlsx_to_pipe_string(csv_path)
            if pipe_data is None:
                continue

            # Read the associated questions
            questions_file = os.path.join(questions_folder, filename.replace(".xlsx", ".json"))
            if not os.path.exists(questions_file):
                print(f"No question file found for {filename}, skipping.")
                continue
            
            questions = read_questions(questions_file)
            
            # Call Gemini API with the processed data and questions
            gemini_response = classify_with_gemini(base_prompt, pipe_data, questions)

            gemini_response = gemini_response.replace("```", "").replace("json", "")

            
            try:
                gemini_response = json.loads(gemini_response)
            except json.JSONDecodeError:
                print(f"Error decoding response for {filename}")
                continue
            
            # Save response as JSON
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(gemini_response, f, indent=4)
            print("Processed ",output_file_path)

if __name__ == "__main__":
    process_csv_files()