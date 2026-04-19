import os
import json
import base64
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# GitHub-ready paths (resolved from repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
image_folder = os.path.join(ROOT_DIR, "MMTabReal", "Table_Images")
questions_folder = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
answer_path = os.path.join(ROOT_DIR, "MMTabReal", "Results", "table-as-image-gpt")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "table-as-image.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")

# Ensure the output folder exists
os.makedirs(answer_path, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Table-as-Image GPT baseline runner")
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

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to read questions
def read_questions(questions_file):
    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [list(q.values())[0] for q in data]  # Extract only question text


def find_first_image(folder_path):
    for root, _, files in os.walk(folder_path):
        for file_name in sorted(files):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                return os.path.join(root, file_name)
    return None

# Function to classify with GPT-4 Vision
def classify_with_gpt(base_prompt, image_path, questions):
    base64_image = encode_image(image_path)
    prompt = (
        base_prompt
        + "\n\nThese are the questions:\n"
        + json.dumps(questions)
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + " Answer the questions based on the image"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
    )
    
    return response.choices[0].message.content

# Function to process image folders
def process_image_folders(question_mode="all"):
    count = 1
    base_prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)
    for subfolder in os.listdir(image_folder):
        subfolder_path = os.path.join(image_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Skip non-directory files
        
        # Find the first image anywhere in the table folder
        image_path = find_first_image(subfolder_path)
        if not image_path:
            print(f"Skipping {subfolder}, no image found.")
            continue
        
        # Find corresponding questions file
        questions_file = os.path.join(questions_folder, subfolder + ".json")
        if not os.path.exists(questions_file):
            print(f"No question file found for {subfolder}, skipping.")
            continue
        
        questions = read_questions(questions_file)
        
        if question_mode == "all":
            gpt_response = classify_with_gpt(base_prompt, image_path, questions)
            count += 1
            
            # Save response as JSON
            output_file_path = os.path.join(answer_path, f"{subfolder}.json")
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump({"responses": gpt_response}, f, indent=4)
            print("Processed", output_file_path)
        else:
            results = []
            for idx, question in enumerate(questions, start=1):
                gpt_response = classify_with_gpt(base_prompt, image_path, [question])
                results.append({f"Question {idx}": question, f"Answer {idx}": gpt_response})
                count += 1
            
            output_file_path = os.path.join(answer_path, f"{subfolder}.json")
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print("Processed", output_file_path)

if __name__ == "__main__":
    args = parse_args()
    process_image_folders(args.question_mode)
