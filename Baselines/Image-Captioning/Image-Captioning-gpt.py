import base64
import io
import json
import os
import argparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES_DIR = os.path.join(ROOT_DIR, "MMTabReal", "all")
QUESTIONS_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
OUTPUT_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Results", "image-captioning-gpt")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "captioning.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Image-captioning GPT baseline runner")
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


def compress_and_encode_image(img_path, quality=30):
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGB")
    img_io = io.BytesIO()
    img.save(img_io, format="JPEG", quality=quality)
    img_io.seek(0)
    return f"data:image/jpeg;base64,{base64.b64encode(img_io.read()).decode('utf-8')}"


def extract_questions(questions_json_path):
    with open(questions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for item in data:
        for key, value in item.items():
            if str(key).lower().startswith("question"):
                questions.append(value)
    return questions


def build_multimodal_content(soup, current_dir):
    table = soup.find("table")
    if not table:
        return [{"type": "text", "text": "No table found in the HTML content."}]

    content = []
    for row in table.find_all("tr"):
        for col in row.find_all(["td", "th"]):
            img_tag = col.find("img")
            if img_tag and "src" in img_tag.attrs:
                img_path = os.path.join(current_dir, img_tag["src"])
                encoded_img = compress_and_encode_image(img_path)
                if encoded_img:
                    content.append({"type": "image_url", "image_url": {"url": encoded_img}})

            text = col.get_text(strip=True)
            if text:
                content.append({"type": "text", "text": text})
            else:
                content.append({"type": "text", "text": "-"})

        content.append({"type": "text", "text": "\n"})

    return content


def find_html_files(folder_path):
    html_files = []
    for subdir in next(os.walk(folder_path))[1]:
        subdir_path = os.path.join(folder_path, subdir)
        for file_name in os.listdir(subdir_path):
            if file_name.endswith(".html"):
                html_files.append(os.path.join(subdir_path, file_name))
    return sorted(html_files)


def process_html_content(base_prompt, question_mode):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        table_content = build_multimodal_content(soup, subdir)

        if question_mode == "all":
            prompt_content = [
                {"type": "text", "text": base_prompt},
                {"type": "text", "text": "\nTable:\n"},
                *table_content,
                {"type": "text", "text": "\nQuestions:\n" + json.dumps(questions)},
            ]
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_content}],
                    max_tokens=700,
                    temperature=0,
                )
                answer_text = response.choices[0].message.content.strip()
                cleaned = answer_text.replace("```json", "").replace("```", "").strip()
                try:
                    payload = json.loads(cleaned)
                except json.JSONDecodeError:
                    payload = {"responses": answer_text}
            except Exception as e:
                payload = {"error": str(e)}
        else:
            payload = []
            for idx, question in enumerate(questions, start=1):
                prompt_content = [
                    {"type": "text", "text": base_prompt},
                    {"type": "text", "text": "\nTable:\n"},
                    *table_content,
                    {"type": "text", "text": "\nQuestion:\n" + question},
                ]
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt_content}],
                        max_tokens=400,
                        temperature=0,
                    )
                    answer_text = response.choices[0].message.content.strip()
                    payload.append({f"Question {idx}": question, f"Answer {idx}": answer_text})
                except Exception as e:
                    payload.append({f"Question {idx}": question, f"Answer {idx}": f"Error: {e}"})

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    args = parse_args()
    load_dotenv()
    prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)
    process_html_content(prompt, args.question_mode)