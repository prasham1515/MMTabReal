import json
import os
import argparse

import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES_DIR = os.path.join(ROOT_DIR, "MMTabReal", "all")
QUESTIONS_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
OUTPUT_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Results", "image-captioning-gemini")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "captioning.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Image-captioning Gemini baseline runner")
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


def extract_questions(questions_json_path):
    with open(questions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for item in data:
        for key, value in item.items():
            if str(key).lower().startswith("question"):
                questions.append(value)
    return questions


def build_text_and_images(soup, current_dir):
    table = soup.find("table")
    if not table:
        return "No table found in the HTML content.", []

    lines = []
    images = []
    image_index = 1

    for row in table.find_all("tr"):
        row_cells = []
        for col in row.find_all(["td", "th"]):
            img_tag = col.find("img")
            text = col.get_text(strip=True)

            if img_tag and "src" in img_tag.attrs:
                img_path = os.path.join(current_dir, img_tag["src"])
                if os.path.exists(img_path):
                    try:
                        images.append(Image.open(img_path).convert("RGB"))
                        row_cells.append(f"[IMAGE_{image_index}]")
                        image_index += 1
                    except Exception:
                        row_cells.append("[MISSING_IMAGE]")

            row_cells.append(text if text else "-")

        lines.append("|".join(row_cells) if row_cells else "-")

    return "\n".join(lines), images


def find_html_files(folder_path):
    html_files = []
    for subdir in next(os.walk(folder_path))[1]:
        subdir_path = os.path.join(folder_path, subdir)
        for file_name in os.listdir(subdir_path):
            if file_name.endswith(".html"):
                html_files.append(os.path.join(subdir_path, file_name))
    return sorted(html_files)


def process_html_content(base_prompt, question_mode):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = genai.GenerativeModel("gemini-2.0-flash")

    for html_file in find_html_files(TABLES_DIR):
        subdir = os.path.dirname(html_file)
        table_name = os.path.basename(subdir)
        questions_file = os.path.join(QUESTIONS_DIR, f"{table_name}.json")
        output_file = os.path.join(OUTPUT_DIR, f"{table_name}.json")

        if os.path.exists(output_file):
            print(f"Skipping {table_name}, results already exist.")
            continue
        if not os.path.exists(questions_file):
            print(f"Skipping {table_name}, questions file not found.")
            continue

        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        questions = extract_questions(questions_file)
        table_text, images = build_text_and_images(soup, subdir)

        if question_mode == "all":
            prompt = (
                base_prompt
                + "\n\nTable (pipe-separated):\n"
                + table_text
                + "\n\nQuestions:\n"
                + json.dumps(questions)
                + "\n\nImages are attached in the same order as IMAGE_1, IMAGE_2, ..."
            )

            payload = [prompt]
            payload.extend(images)

            try:
                response = model.generate_content(payload)
                answer_text = (response.text or "").strip()
                cleaned = answer_text.replace("```json", "").replace("```", "").strip()
                try:
                    result = json.loads(cleaned)
                except json.JSONDecodeError:
                    result = {"responses": answer_text}
            except Exception as e:
                result = {"error": str(e)}
        else:
            result = []
            for idx, question in enumerate(questions, start=1):
                prompt = (
                    base_prompt
                    + "\n\nTable (pipe-separated):\n"
                    + table_text
                    + "\n\nQuestion:\n"
                    + question
                    + "\n\nImages are attached in the same order as IMAGE_1, IMAGE_2, ..."
                )
                payload = [prompt]
                payload.extend(images)
                try:
                    response = model.generate_content(payload)
                    answer_text = (response.text or "").strip()
                    result.append({f"Question {idx}": question, f"Answer {idx}": answer_text})
                except Exception as e:
                    result.append({f"Question {idx}": question, f"Answer {idx}": f"Error: {e}"})

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    args = parse_args()
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)
    process_html_content(prompt, args.question_mode)