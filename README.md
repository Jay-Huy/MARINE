# MARINE Project

This repository contains the codebase for running the MARINE (Masked Attention Regularization in Natural Environments) pipeline on the LLaVA framework. 

## Structure
- `marine/generate_llava2.py`: Main script for generating evaluation outputs using the Double Pass (Spatial CFG + Dynamic Gamma).
- `marine/utils/utils_dataset.py`: Handles data loading and SAM masking integration.
- `marine/utils/utils_guidance.py`: Soft Spatial Penalty via Pre-hook (`torch.log(alpha)`).

## Requirements & Setup
Make sure you have LLaVA and its dependencies installed in your environment.

### Preparing the Data
Before running the evaluation on Colab or Kaggle, you **must** place your image dataset and SAM mask arrays in matching paths:

1. **Images Folder (`--image_folder`)**:
   Contains the raw `.jpg` or `.png` images.
   Example: `data/coco_images`

2. **SAM Mask Arrays**:
   This folder **must be located in the same parent directory** as your `image_folder`, and must be named `sam3_mask_arrays`.
   Example:
   ```text
   data/
   ├── coco_images/
   │   └── COCO_val2014_000000144305.jpg
   └── sam3_mask_arrays/
       └── COCO_val2014_000000144305/
           └── masks.npy
   ```
   *Note: The dataloader uses `os.path.dirname(image_folder)` to automatically find `sam3_mask_arrays`. Ensure this structure is strictly followed!*

## Running the Pipeline

### 1. Test Mode (Quick Validation)
You can run a quick test using the `--test_samples` flag to limit execution to 10 samples. This is highly recommended before a full execution run.

```bash
!PYTHONPATH=/content/LLaVA python -m marine.generate_llava2 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --question_path ./data/marine_qa/question \
    --question_file chair_coco_detr_th0.95_ram_th0.68.json \
    --image_folder /content/drive/MyDrive/AI-Algorithm/Final-Project/data/coco_images \
    --answer_path ./output \
    --answers_file marine_eval_chair_alpha0.7.json \
    --decode_approach 2 \
    --alpha 0.7 \
    --tau 2.8 \
    --beta 3.0 \
    --batch_size 2 \
    --load_4bit \
    --test_samples 10
```

### 2. Full Execution (Generation)
To run the full evaluation over the entire dataset, simply omit the `--test_samples` flag. The script will output total processed samples and execution time at the end.

```bash
!PYTHONPATH=/content/LLaVA python -m marine.generate_llava2 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --question_path ./data/marine_qa/question \
    --question_file chair_coco_detr_th0.95_ram_th0.68.json \
    --image_folder /content/drive/MyDrive/AI-Algorithm/Final-Project/data/coco_images \
    --answer_path ./output \
    --answers_file marine_eval_chair_alpha0.7.json \
    --decode_approach 2 \
    --alpha 0.7 \
    --tau 2.8 \
    --beta 3.0 \
    --batch_size 2 \
    --load_4bit
```

**Results:** The final output will be saved as a standard `.json` file containing both metadata (model parameters, execution time) and the generated responses list.

## Ablation Study Scripts

These scripts are pre-configured to run different ablation studies to isolate the impact of various MARINE components.

### Ablation 1: Masking Only (No CFG / No Dynamic Gamma)
Demonstrates why Classifier-Free Guidance Theory is necessary by forcing the model to generate directly from the masked image.
```bash
!PYTHONPATH=/content/LLaVA python -m marine.generate_llava2 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --question_path ./data/marine_qa/question \
    --question_file chair_coco_detr_th0.95_ram_th0.68.json \
    --image_folder /content/drive/MyDrive/AI-Algorithm/Final-Project/data/coco_images \
    --answer_path ./output \
    --answers_file ablation_mask_only.json \
    --decode_approach 1 \
    --batch_size 2 \
    --load_4bit
```

### Ablation 2: Static Gamma (No Dynamic Entropy)
Fixes the gamma parameter to a static value (0.7) to prove that dynamically shifting gamma based on entropy yields better contextual fluency.
```bash
!PYTHONPATH=/content/LLaVA python -m marine.generate_llava2 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --question_path ./data/marine_qa/question \
    --question_file chair_coco_detr_th0.95_ram_th0.68.json \
    --image_folder /content/drive/MyDrive/AI-Algorithm/Final-Project/data/coco_images \
    --answer_path ./output \
    --answers_file ablation_static_gamma0.7.json \
    --decode_approach 2 \
    --static_gamma 0.7 \
    --alpha 0.7 \
    --batch_size 2 \
    --load_4bit
```

### Ablation 3: Hard Spatial Mask (Alpha = 0.0)
Completely removes the general background (objects are blacked out) instead of softly dimming it.
```bash
!PYTHONPATH=/content/LLaVA python -m marine.generate_llava2 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --question_path ./data/marine_qa/question \
    --question_file chair_coco_detr_th0.95_ram_th0.68.json \
    --image_folder /content/drive/MyDrive/AI-Algorithm/Final-Project/data/coco_images \
    --answer_path ./output \
    --answers_file ablation_hard_mask_alpha0.0.json \
    --decode_approach 2 \
    --alpha 0.0 \
    --tau 2.8 \
    --beta 3.0 \
    --batch_size 2 \
    --load_4bit
```
