import argparse
import json
import os

import torch
from bs4 import BeautifulSoup
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES_DIR = os.path.join(ROOT_DIR, "MMTabReal", "all")
QUESTIONS_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
OUTPUT_BASE_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Results")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "captioning.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")

MODEL_IDS = {
    "mantis": "TIGER-Lab/Mantis-8B-Idefics2",
    "phi": "microsoft/Phi-3.5-vision-instruct",
    "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
}

GENERATION_KWARGS = {
    "max_new_tokens": 1024,
    "num_beams": 1,
    "do_sample": False,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Image-captioning HF baseline runner")
    parser.add_argument(
        "--model",
        default="mantis",
        choices=["mantis", "phi", "qwen25", "qwen3"],
        help="Model backend to use",
    )
    parser.add_argument(
        "--question-mode",
        default="all",
        choices=["all", "one-by-one"],
        help="Ask all questions in one request or ask one question per request",
    )
    return parser.parse_args()


def load_model_and_processor(model_name):
    model_id = MODEL_IDS[model_name]

    if model_name == "mantis":
        model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor

    if model_name == "phi":
        device_map = "cuda" if torch.cuda.is_available() else "auto"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation="flash_attention_2",
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation="eager",
            )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor

    if model_name == "qwen25":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor

    if model_name == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor

    raise ValueError(f"Unsupported model: {model_name}")


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


def find_html_files(folder_path):
    html_files = []
    for subdir in next(os.walk(folder_path))[1]:
        subdir_path = os.path.join(folder_path, subdir)
        for file_name in os.listdir(subdir_path):
            if file_name.endswith(".html"):
                html_files.append(os.path.join(subdir_path, file_name))
    return sorted(html_files)


def build_message_blocks(soup, current_dir, base_prompt, max_images=30):
    table = soup.find("table")
    content = [{"type": "text", "text": base_prompt.strip() + "\n\nTable:\n"}]
    if not table:
        content.append({"type": "text", "text": "No table found in HTML."})
        return content

    image_count = 0
    for row in table.find_all("tr"):
        for col in row.find_all(["td", "th"]):
            content.append({"type": "text", "text": "|"})
            img_tag = col.find("img")
            if img_tag and "src" in img_tag.attrs:
                img_path = os.path.join(current_dir, img_tag["src"])
                if os.path.exists(img_path) and image_count < max_images:
                    try:
                        content.append({
                            "type": "image",
                            "image": Image.open(img_path).convert("RGB"),
                        })
                        image_count += 1
                    except Exception:
                        content.append({"type": "text", "text": "[MISSING_IMAGE]"})
                else:
                    content.append({"type": "text", "text": "[MISSING_IMAGE]"})

            text = col.get_text(strip=True)
            content.append({"type": "text", "text": text if text else "-"})

        content.append({"type": "text", "text": "|\n"})

    return content


def main():
    args = parse_args()
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"image-captioning-hf-{args.model}")
    os.makedirs(output_dir, exist_ok=True)

    base_prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)
    model, processor = load_model_and_processor(args.model)
    model_device = next(model.parameters()).device

    for html_file in find_html_files(TABLES_DIR):
        subdir = os.path.dirname(html_file)
        table_name = os.path.basename(subdir)
        questions_file = os.path.join(QUESTIONS_DIR, f"{table_name}.json")
        result_path = os.path.join(output_dir, f"{table_name}.json")

        if os.path.exists(result_path):
            print(f"Skipping {table_name}, already processed.")
            continue
        if not os.path.exists(questions_file):
            print(f"Skipping {table_name}, no questions file found.")
            continue

        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        questions = extract_questions(questions_file)
        table_blocks = build_message_blocks(soup, subdir, base_prompt)
        messages = [{"role": "user", "content": table_blocks}]

        results = []
        if args.question_mode == "all":
            messages_all = messages + [{
                "role": "user",
                "content": [{"type": "text", "text": "\nQuestions:\n" + json.dumps(questions)}],
            }]

            prompt = processor.apply_chat_template(messages_all, add_generation_prompt=True)
            images = [block["image"] for m in messages_all for block in m["content"] if block["type"] == "image"]

            try:
                inputs = processor(text=prompt, images=images if images else None, return_tensors="pt")
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(**inputs, **GENERATION_KWARGS)
                trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
                decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                results = {"responses": decoded.strip()}
            except Exception as e:
                results = {"error": f"Failed in all-questions mode: {e}"}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            for idx, question in enumerate(questions, start=1):
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "\nQuestion: " + question}],
                })

                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                images = [block["image"] for m in messages for block in m["content"] if block["type"] == "image"]

                try:
                    inputs = processor(text=prompt, images=images if images else None, return_tensors="pt")
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                except Exception as e:
                    print(f"Failed to build model inputs for {table_name} Q{idx}: {e}")
                    continue

                with torch.no_grad():
                    output_ids = model.generate(**inputs, **GENERATION_KWARGS)

                trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
                decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                results.append({f"Question {idx}": question, f"Answer {idx}": decoded.strip()})

                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": decoded.strip()}],
                })

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {result_path}")


if __name__ == "__main__":
    main()