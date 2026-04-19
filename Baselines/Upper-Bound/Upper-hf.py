import os
import json
import argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# GitHub-ready paths (resolved from repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Upper_Bound")
QUESTIONS_DIR = os.path.join(ROOT_DIR, "MMTabReal", "Questions")
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "upper.txt")
FORMAT_FILE = os.path.join(PROMPTS_DIR, "formatting_guidelines.txt")

MODEL_IDS = {
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "llama": "meta-llama/Meta-Llama-3-8B",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "phi": "microsoft/Phi-3.5-vision-instruct",
    "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
}

GENERATION_KWARGS = {
    "max_new_tokens": 1024,
    "do_sample": False,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Upper-bound HF baseline runner")
    parser.add_argument(
        "--model",
        default="mixtral",
        choices=["mixtral", "llama", "llama3", "phi", "qwen25", "qwen3"],
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

    if model_name in {"mixtral", "llama", "llama3"}:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
        )
        return model, tokenizer, False

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
        return model, processor, True

    if model_name == "qwen25":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, True

    if model_name == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, True

    raise ValueError(f"Unsupported model: {model_name}")


def read_questions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [list(item.values())[0] for item in data]


def load_prompt_text(prompt_path, format_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    with open(format_path, "r", encoding="utf-8") as f:
        formatting = f.read().strip()
    return prompt_text.replace("{ANSWER_FORMATTING_GUIDELINES}", formatting)


def xlsx_to_pipe_string(xlsx_path):
    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        print(f"[!] Error reading {xlsx_path}: {e}")
        return None

    parts = []
    for sheet_name, df in sheets.items():
        df = df.map(lambda x: ''.join(ch for ch in str(x) if ord(ch) < 128) if isinstance(x, str) else x)
        parts.append(f"[Sheet: {sheet_name}]\n{df.to_csv(sep='|', index=False)}")
    return "\n".join(parts)


def make_prompt(base_prompt, table_psv, questions):
    return (
        base_prompt
        + "\n\nHere is the dataset in a pipe-separated format:\n"
        + table_psv
        + "\n\nAnswer the following questions based on the dataset:\n"
        + json.dumps(questions)
    )


def generate_answer(model, processor, prompt, multimodal=False):
    if multimodal:
        try:
            inputs = processor(text=prompt, return_tensors="pt")
        except TypeError:
            inputs = processor(prompt, return_tensors="pt")
    else:
        inputs = processor(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, **GENERATION_KWARGS)

    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    args = parse_args()
    output_dir = os.path.join(ROOT_DIR, "MMTabReal", "Results", f"upper-hf-{args.model}")
    os.makedirs(output_dir, exist_ok=True)
    base_prompt = load_prompt_text(PROMPT_FILE, FORMAT_FILE)

    model, processor, multimodal = load_model_and_processor(args.model)

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".xlsx"):
            continue

        xlsx_path = os.path.join(DATA_DIR, filename)
        q_path = os.path.join(QUESTIONS_DIR, filename.replace(".xlsx", ".json"))
        out_path = os.path.join(output_dir, filename.replace(".xlsx", ".json"))

        if not os.path.isfile(q_path):
            print(f"[-] Skipping {filename}: no questions JSON")
            continue
        if os.path.exists(out_path):
            print(f"[=] Already done {filename}, skipping")
            continue

        table_psv = xlsx_to_pipe_string(xlsx_path)
        if table_psv is None:
            continue

        questions = read_questions(q_path)
        prompt = make_prompt(base_prompt, table_psv, questions)

        print(f"[->] Asking HF model ({args.model}) for {filename} ...")
        answer_text = generate_answer(model, processor, prompt, multimodal=multimodal)

        answer_json = answer_text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(answer_json)
        except json.JSONDecodeError:
            parsed = answer_text

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved answer to {out_path}")


if __name__ == "__main__":
    main()