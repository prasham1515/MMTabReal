import os
import json
import argparse
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

# GitHub-ready paths (resolved from repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "MMTabReal", "all")
QUESTIONS_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "interleaved.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")

MODEL_IDS = {
    "mantis": "TIGER-Lab/Mantis-8B-Idefics2",
    "phi": "microsoft/Phi-3.5-vision-instruct",
    "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
}

generation_kwargs = {
    "max_new_tokens": 1024,
    "num_beams": 1,
    "do_sample": False
}


def parse_args():
    parser = argparse.ArgumentParser(description="Interleaved HF baseline runner")
    parser.add_argument(
        "--model",
        default="mantis",
        choices=["mantis", "phi", "qwen25", "qwen3"],
        help="Model backend to use",
    )
    parser.add_argument(
        "--question-mode",
        default="one-by-one",
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
            # Fallback for environments without flash-attn.
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

# Convert CSV to multimodal message blocks
def build_message_blocks(df, image_dir, base_prompt, max_images=30):
    content = [{"type": "text", "text": base_prompt.strip()}]
    image_count = 0
    for _, row in df.iterrows():
        for cell in row:
            cell_str = str(cell).strip()
            if cell_str.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(image_dir, cell_str)
                if os.path.exists(img_path) and image_count < max_images:
                    content.append({"type": "text", "text": "|"})
                    content.append({
                        "type": "image",
                        "image": Image.open(img_path).convert("RGB")
                    })
                    image_count += 1
                else:
                    content.append({"type": "text", "text": "|[MissingImage]"})
            else:
                content.append({"type": "text", "text": f"|{cell_str}"})
        content.append({"type": "text", "text": "|"})
        content.append({"type": "text", "text": "\n"})
    return content

def main():
    args = parse_args()
    output_dir = os.path.join(ROOT_DIR, "MMTabReal", "Results", f"interleaved-hf-{args.model}")
    os.makedirs(output_dir, exist_ok=True)
    base_prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)

    model, processor = load_model_and_processor(args.model)

    # Main loop
    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        result_path = os.path.join(output_dir, f"{folder_name}.json")
        print(result_path)
        if os.path.exists(result_path):
            print(f"Skipping {folder_name}, already processed.")
            continue
        if ("pl_table" in result_path):
            continue
        if ("Fifa" in result_path):
            continue
        if ("Kenshi" in result_path):
            continue

        csv_path = os.path.join(folder_path, f"{folder_name}.csv")
        images_dir = os.path.join(folder_path, "images")
        questions_path = os.path.join(QUESTIONS_DIR, f"{folder_name}.json")

        if not os.path.exists(csv_path) or not os.path.exists(questions_path):
            print(f"Missing files for {folder_name}")
            continue

        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
        except Exception as e:
            print(f"❌ Failed to read CSV for {folder_name}: {e}")
            continue

        with open(questions_path, "r", encoding="utf-8") as qf:
            qdata = json.load(qf)
            questions = [v for item in qdata for k, v in item.items() if k.lower().startswith("question")]

        # Build table context
        table_blocks = build_message_blocks(df, images_dir, base_prompt)
        messages = [{"role": "user", "content": table_blocks}]

        results = []

        for idx, question in enumerate(questions, start=1):
            print(f"🔍 Q{idx}: {question}")
            # Add question to conversation
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": question}]
            })

            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            images = [block["image"] for m in messages for block in m["content"] if block["type"] == "image"]
            image_input = images if images else None

            try:
                inputs = processor(text=prompt, images=image_input, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            except Exception as e:
                print(f"❌ Failed to prepare inputs for Q{idx} in {folder_name}: {e}")
                continue

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)

            trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            print(f"✅ A{idx}: {decoded}")
            results.append({f"Question {idx}": question, f"Answer {idx}": decoded})

            # Add assistant response to history
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": decoded}]
            })

            torch.cuda.empty_cache()

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ All responses saved to {result_path}")


if __name__ == "__main__":
    main()
