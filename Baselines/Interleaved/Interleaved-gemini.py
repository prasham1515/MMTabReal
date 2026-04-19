import base64
import os
import json
import io
import argparse
from bs4 import BeautifulSoup
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import sys

# GitHub-ready paths (resolved from repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES_DIR = os.path.join(ROOT_DIR, "MMTabReal", "all")
QUESTIONS_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
OUTPUT_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Results", "interleaved-gemini")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "interleaved.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")


def load_prompt_text(prompt_path, format_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    with open(format_path, "r", encoding="utf-8") as f:
        formatting = f.read().strip()
    return prompt_text.replace("{ANSWER_FORMATTING_GUIDELINES}", formatting)

def parse_args():
    parser = argparse.ArgumentParser(description="Interleaved Gemini baseline runner")
    parser.add_argument(
        "--question-mode",
        default="all",
        choices=["all", "one-by-one"],
        help="Ask all questions in one request or ask one question per request",
    )
    return parser.parse_args()

def compress_and_encode_image(img_path, quality=30):
    """
    Compress an image, encode it in base64, and return the base64 string.
    """
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return None
    
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_io = io.BytesIO()
    img.save(img_io, format="JPEG", quality=quality)
    img_io.seek(0)
    return f"data:image/jpeg;base64,{base64.b64encode(img_io.read()).decode('utf-8')}"

def convert_content(soup, current_dir):
    table = soup.find('table')
    if not table:
        return "No table found in the HTML content"
    
    content = []
    for row in table.find_all('tr'):
        for col in row.find_all(['td', 'th']):
            img_tag = col.find('img')
            if img_tag and 'src' in img_tag.attrs:
                img_path = os.path.join(current_dir, img_tag['src'])
                encoded_img = compress_and_encode_image(img_path)
                if encoded_img:
                    content.append({"type": "image_url", "image_url": {"url": encoded_img}})
            text = col.get_text(strip=True) or '-'
            if text:
                content.append({"type": "text", "text": text})
    
    return content

def process_html_content(folder_path, questions_path, api_keys, output_path, base_prompt, question_mode="all"):
    client = OpenAI(api_key=api_keys)
    key_index = 0
    
    html_files = []
    for subdir in next(os.walk(folder_path))[1]:
        subdir_path = os.path.join(folder_path, subdir)
        for file in os.listdir(subdir_path):
            if file.endswith('.html'):
                html_files.append(os.path.join(subdir_path, file))
    
    for html_file in html_files:
        subdir = os.path.dirname(html_file)
        table_name = os.path.basename(subdir)
        json_file = os.path.join(questions_path, f"{table_name}.json")
        output_file = os.path.join(output_path, os.path.basename(subdir) + '.json')
        if os.path.exists(output_file):
            print(f"Skipping {subdir}, results already exist.")
            continue
        if not os.path.exists(json_file):
            continue
        
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        converted_content = convert_content(soup, subdir)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        question_list = [q_data[next(iter(q_data))] for q_data in questions if 'Question' in next(iter(q_data))]
        prompt_content = [{"type": "text", "text": base_prompt}] + converted_content + \
                         [{"type": "text", "text": "Questions: " + json.dumps(question_list)}]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=500
            )
            response_data = response.choices[0].message.content.strip()
            
            # Ensure JSON parsing safety
            try:
                response_data = json.loads(response_data)
            except json.JSONDecodeError:
                response_data = [{"Error": "Invalid JSON response from OpenAI", "RawResponse": response_data}]
        except Exception as e:
            response_data = [{"Error": str(e)}]
        print(response_data)
        if response_data[0]['RawResponse']:
            response_data=response_data[0]['RawResponse']
        response_data = response_data.replace("```", "").replace("json", "").replace("\n","")

        output_file = os.path.join(output_path, os.path.basename(subdir) + '.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=4)
            print("Saved at ",output_file)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_dotenv()
    api_keys = os.getenv("OPENAI_API_KEY")
    base_prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)
    process_html_content(TABLES_DIR, QUESTIONS_DIR, api_keys, OUTPUT_DIR, base_prompt, args.question_mode)
