import os
import json
import argparse
import torch
import gc
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
)

# GitHub-ready paths (resolved from repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
QUESTIONS_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
TABLE_IMAGES_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Table_Images")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "table-as-image.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")

MODEL_IDS = {
    "table_llava": "SpursgoZmy/table-llava-v1.5-7b-hf",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "mantis": "TIGER-Lab/Mantis-8B-Idefics2",
    "phi": "microsoft/Phi-3.5-vision-instruct",
    "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
    "intern": "OpenGVLab/InternVL2_5-8B",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Table-as-Image HF baseline runner")
    parser.add_argument(
        "--model",
        default="table_llava",
        choices=["table_llava", "llava", "mantis", "phi", "qwen25", "qwen3", "intern"],
        help="Vision-language backend to use",
    )
    parser.add_argument(
        "--question-mode",
        default="one-by-one",
        choices=["all", "one-by-one"],
        help="Ask all questions in one request or ask one question per request",
    )
    return parser.parse_args()


OUTPUT_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Results", "table-as-image")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def load_prompt_text(prompt_path, format_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    with open(format_path, "r", encoding="utf-8") as f:
        formatting = f.read().strip()
    return prompt_text.replace("{ANSWER_FORMATTING_GUIDELINES}", formatting)

def parse_questions(questions_file):
    questions_content = []
    with open(questions_file, "r", encoding="utf-8") as qf:
        questions_data = json.load(qf)
        for entry in questions_data:
            for key, value in entry.items():
                if key.startswith("Question"):
                    questions_content.append(value)
    return questions_content


def find_first_image(folder_path):
    image_candidates = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                image_candidates.append(os.path.join(root, file_name))
    if not image_candidates:
        return None
    return sorted(image_candidates)[0]


def load_model_and_processor(model_name):
    model_id = MODEL_IDS[model_name]

    if model_name == "table_llava":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, "llava"

    if model_name == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, "llava"

    if model_name == "mantis":
        model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, "generic"

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
        return model, processor, "generic"

    if model_name == "qwen25":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, "generic"

    if model_name == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, "generic"

    if model_name == "intern":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor, "generic"

    raise ValueError(f"Unsupported model: {model_name}")


def build_inputs(processor, image, question, prompt_style, base_prompt):
    if prompt_style == "llava" and hasattr(processor, "apply_chat_template"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": base_prompt + "\nThis is the image <|image_1|>.\nQuestion: " + question},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return processor(images=image, text=prompt, return_tensors="pt"), conversation

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": base_prompt + "\nQuestion: " + question},
                {"type": "image"},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(images=image, text=prompt, return_tensors="pt"), messages

    return processor(text=base_prompt + "\nQuestion: " + question, images=image, return_tensors="pt"), messages


def decode_output(processor, output_ids):
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if hasattr(processor, "decode"):
        return processor.decode(output_ids[0], skip_special_tokens=True)
    return str(output_ids)


def main():
    args = parse_args()
    output_dir = os.path.join(ROOT_DIR, "MMTabReal", "Results", f"table-as-image-{args.model}")
    os.makedirs(output_dir, exist_ok=True)
    base_prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)

    model, processor, prompt_style = load_model_and_processor(args.model)
    model_device = next(model.parameters()).device

    # Process Folders
    for folder in os.listdir(TABLE_IMAGES_DIR):
        table_images_folder_path = os.path.join(TABLE_IMAGES_DIR, folder)
        if not os.path.isdir(table_images_folder_path):
            continue

        result_file = os.path.join(output_dir, f"{folder}.json")
        if os.path.exists(result_file):
            print(f"Skipping {folder}, results already exist.")
            continue

        questions_file = os.path.join(QUESTIONS_DIR, f"{folder}.json")
        if not os.path.exists(questions_file):
            print(f"Skipping {folder}, no questions file found.")
            continue

        image_file_path = find_first_image(table_images_folder_path)
        if not image_file_path:
            print(f"Skipping {folder}, no image found.")
            continue

        print(image_file_path)
        image = Image.open(image_file_path).convert("RGB")
        questions_content = parse_questions(questions_file)

        results = []
        for idx, question in enumerate(questions_content, start=1):
            inputs, conversation = build_inputs(processor, image, question, prompt_style, base_prompt)
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

            prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            response_text = decode_output(processor, output[:, prompt_len:])

            marker = "ASSISTANT:"
            if marker in response_text:
                final_answer_str = response_text.split(marker)[-1].strip()
            else:
                final_answer_str = response_text.strip()

            try:
                parsed_response = json.loads(final_answer_str)
                if isinstance(parsed_response, list) and len(parsed_response) > 0:
                    answer_entry = parsed_response[0]
                elif isinstance(parsed_response, dict):
                    answer_entry = parsed_response
                else:
                    answer_entry = {"question": question, "answer": final_answer_str}
            except Exception as e:
                print(f"Error parsing JSON for question: {question}. Exception: {e}")
                answer_entry = {"question": question, "answer": final_answer_str}

            results.append(answer_entry)

            del inputs, output, conversation
            gc.collect()
            torch.cuda.empty_cache()

        output_file = os.path.join(output_dir, f"{folder}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
