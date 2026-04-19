# MMTabReal

MMTabReal is a real-world benchmark for multimodal table understanding. It contains 500 human-curated tables paired with 4,021 question-answer pairs and focuses on multimodal tables that mix text with charts, maps, icons, logos, color encodings, and other visual cues. The benchmark spans four question types, five reasoning categories, and eight structural archetypes, and it is released for evaluation only.

This repository is organized like a research-code release: dataset artifacts, prompt templates, baseline implementations (GPT, Gemini, HF), and evaluation scripts are separated for reproducibility.

## Dataset Download

In order to download the dataset run 
```bash
bash Dataset/download_dataset.sh
```

## Repository Structure (after dataset download)

```text
MMTabReal/
├── Baselines/
│   ├── Image-Captioning/
│   │   ├── Image-Captioning-gemini.py
│   │   ├── Image-Captioning-gpt.py
│   │   └── Image-Captioning-hf.py
│   ├── Interleaved/
│   │   ├── Interleaved-gemini.py
│   │   ├── Interleaved-gpt.py
│   │   └── interleaved-hf.py
│   ├── Missing-Image/
│   │   ├── Lower-gemini.py
│   │   ├── Lower-gpt.py
│   │   └── Lower-hf.py
│   ├── Table-As-Image/
│   │   ├── Table-as-Image-gemini.py
│   │   ├── Table-as-Image-gpt.py
│   │   └── Table-as-Image-hf.py
│   └── Upper-Bound/
│       ├── Upper-gemini.py
│       ├── Upper-gpt.py
│       └── Upper-hf.py
├── Dataset/
│   └── download_dataset.sh
├── Eval/
│   └── eval.py
├── MMTabReal/
│   ├── Question-Metadata/
│   ├── Questions/
│   ├── Table_Images/
│   ├── Upper_Bound/
│   └── all/
├── Prompts/
│   ├── captioning.txt
│   ├── formatting_guidelines.txt
│   ├── interleaved.txt
│   ├── missing-image.txt
│   ├── table-as-image.txt
│   └── upper.txt
├── Utils/
│   ├── convert.py
│   ├── htlm_to_csv.py
│   ├── html_to_image.py
│   ├── json_html.py
│   ├── preprocess copy.py
│   └── preprocess.py
└── ReadME.md
```

## Baselines

### 1) Interleaved
The original multimodal table is preserved, with images embedded inside the table. This setting tests joint reasoning over text and visual information.

### 2) Missing-Image (Lower Bound)
All images are removed from the table, so the model must infer missing visual information from the remaining text. This serves as the lower-performance bound.

### 3) Entity Replaced (Upper Bound)
Images are replaced with precise textual descriptions so the table becomes fully informative. This serves as the upper-performance bound.

### 4) Table-As-Image
The entire table is rendered as an image, requiring the model to interpret structure, text, and visuals from a single visual input.

### 5) Image-Captioning
Images are captioned automatically, and the resulting captions are inserted back into the table before answering.


## Setup

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure API keys

Create a .env file in the repository root:

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

## Running Baselines

### Supported Arguments

- `--model` (HF baselines only): Vision-language model backend
- `--question-mode` (all baselines): Question batching strategy
  - `all`: Send all questions in one API call (faster)
  - `one-by-one`: Process each question separately (resource-efficient)

### Example Commands

```bash
# GPT baseline (all questions in one call)
python Baselines/Interleaved/Interleaved-gpt.py

# Gemini baseline with one-by-one mode
python Baselines/Missing-Image/Lower-gemini.py --question-mode one-by-one

# HF baseline with specific model
python Baselines/Upper-Bound/Upper-hf.py --model mixtral --question-mode all

# Table-as-Image HF with model selection
python Baselines/Table-As-Image/Table-as-Image-hf.py --model table_llava

# Image-Captioning with one-by-one question mode
python Baselines/Image-Captioning/Image-Captioning-hf.py --model mantis --question-mode one-by-one
```

### Available Models per Task

| Task | HF Models |
|------|----------|
| **Interleaved** | `mantis`, `phi`, `qwen25`, `qwen3` |
| **Missing-Image** | `mixtral`, `llama`, `llama3`, `phi`, `qwen25`, `qwen3` |
| **Upper-Bound** | `mixtral`, `llama`, `llama3`, `phi`, `qwen25`, `qwen3` |
| **Table-As-Image** | `table_llava`, `llava`, `mantis`, `phi`, `qwen25`, `qwen3`, `intern` |
| **Image-Captioning** | `mantis`, `phi`, `qwen25`, `qwen3` |

## Outputs

Each script writes predictions under:

```text
MMTBench/Results/<baseline-name>
```

or model-specific subfolders such as:

```text
MMTBench/Results/interleaved-hf-<model>
MMTBench/Results/upper-hf-<model>
MMTBench/Results/image-captioning-hf-<model>
```

## Evaluation

The evaluation script is available at:

```text
Eval/eval.py
```

It computes lexical and overlap-based metrics (for example exact, substring, F1, BLEU, ROUGE variants) from prediction files against gold answers.

## Reproducibility Notes

- Keep folder names in MMTBench/all aligned with MMTBench/Questions JSON names.
- Ensure API keys are set before running GPT/Gemini scripts.
- Large HF models may require multi-GPU or reduced precision depending on hardware.

## Citation

Copy this box directly:

```bibtex

```

## License

MIT License. See [LICENSE](LICENSE) for details.