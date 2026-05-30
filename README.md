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
python -m marine.generate_llava2 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --question_path ./data/marine_qa/question \
    --question_file chair_coco_detr_th0.95_ram_th0.68.json \
    --image_folder ./data/coco_images \
    --answer_path ./output \
    --answers_file marine_eval_chair_alpha0.7.json \
    --decode_approach 2 \
    --alpha 0.7 \
    --tau 2.8 \
    --beta 3.0 \
    --test_samples 10
```

### 2. Full Execution (Generation)
To run the full evaluation over the entire dataset, simply omit the `--test_samples` flag. The script will output total processed samples and execution time at the end.

```bash
python -m marine.generate_llava2 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --question_path ./data/marine_qa/question \
    --question_file chair_coco_detr_th0.95_ram_th0.68.json \
    --image_folder ./data/coco_images \
    --answer_path ./output \
    --answers_file marine_eval_chair_alpha0.7.json \
    --decode_approach 2 \
    --alpha 0.7 \
    --tau 2.8 \
    --beta 3.0
```

**Results:** The final output will be saved as a standard `.json` file containing both metadata (model parameters, execution time) and the generated responses list.
