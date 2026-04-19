import os
import json
import google.generativeai as genai
import PIL.Image
import time
import argparse
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
# GitHub-ready paths (resolved from repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
image_folder = os.path.join(ROOT_DIR, "MMTabReal", "Table_Images")
questions_folder = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
answer_path = os.path.join(ROOT_DIR, "MMTabReal", "Results", "table-as-image-gemini")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "table-as-image.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")

# Ensure the output folder exists
os.makedirs(answer_path, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Table-as-Image Gemini baseline runner")
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


def find_first_image(folder_path):
    for root, _, files in os.walk(folder_path):
        for file_name in sorted(files):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                return os.path.join(root, file_name)
    return None

def classify_with_gemini(base_prompt, image, questions, question_mode="all"):
    """Send the image along with extracted questions to Gemini API."""
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    if question_mode == "all":
        prompt = base_prompt + "\n\nThese are the questions:\n" + json.dumps(questions)
        response = model.generate_content([prompt, image])
        return response.text
    else:
        results = []
        for idx, question in enumerate(questions, start=1):
            prompt = base_prompt + "\n\nQuestion:\n" + question
            response = model.generate_content([prompt, image])
            results.append({f"Question {idx}": question, f"Answer {idx}": response.text})
        return json.dumps(results)
    

def process_image_folders(question_mode="all"):
    """Process images in subfolders, extract questions, and send to Gemini."""
    count=1
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
        image = PIL.Image.open(image_path)
        
        # Find corresponding questions file
        questions_file = os.path.join(questions_folder, subfolder + ".json")
        if not os.path.exists(questions_file):
            print(f"No question file found for {subfolder}, skipping.")
            continue
        
        questions = read_questions(questions_file)
        if count%15==0:
            time.sleep(60)
        # Call Gemini API with the image and questions
        if question_mode == "all":
            gemini_response = classify_with_gemini(base_prompt, image, questions, "all")
            # Save response as JSON
            output_file_path = os.path.join(answer_path, f"{subfolder}.json")
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump({"responses": gemini_response}, f, indent=4)
            print("Processed", output_file_path)
        else:
            results = json.loads(classify_with_gemini(base_prompt, image, questions, "one-by-one"))
            output_file_path = os.path.join(answer_path, f"{subfolder}.json")
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print("Processed", output_file_path)

if __name__ == "__main__":
    args = parse_args()
    process_image_folders(args.question_mode)
