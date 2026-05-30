import argparse
import torch
import os
import json
import shortuuid
import time

from torch.utils.data import DataLoader
from transformers import LogitsProcessorList

from marine.utils.utils import get_chunk, get_answers_file_name, get_model_name_from_path
from marine.utils.utils_dataset import COCOEvalDataset, custom_collate_fn
from marine.utils.utils_guidance import GuidanceLogits

class AttentionCatcher:
    def __init__(self):
        self.attention = None
    def hook_fn(self, module, input, output):
        # LlamaAttention returns (attn_output, attn_weights, past_key_value)
        # attn_weights is at index 1
        self.attention = output[1]

from marine.utils.utils_model import load_model


def eval_model(args):
    
    # Model
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    
    model, tokenizer, processor = load_model(model_name, model_path)

    # Đăng ký Hook vào Layer cuối cùng của Uncondition model để hứng Attention tính Entropy
    catcher = AttentionCatcher()
    # LLaVA 1.5 dùng kiến trúc LLaMA, layer cuối là model.model.layers[-1].self_attn
    hook_handle = model.model.layers[-1].self_attn.register_forward_hook(catcher.hook_fn)

    # QA Data
    questions = json.load(open(os.path.expanduser(
        os.path.join(args.question_path, args.question_file)), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    if args.test_samples is not None:
        questions = questions[:args.test_samples]
        print(f"[*] TEST MODE: Limited to {args.test_samples} samples.")

    if args.answers_file is None:
        args.answers_file = get_answers_file_name(args, model_name)

    answers_file = os.path.expanduser(
        os.path.join(args.answer_path, args.answers_file))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    dataset = COCOEvalDataset(questions, args.image_folder, processor, tokenizer, args.conv_mode, getattr(model.config, 'mm_use_im_start_end', False))
    eval_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    start_time = time.time()
    
    final_output = {
        "metadata": {
            "model_id": model_name,
            "decode_approach": args.decode_approach,
            "alpha": args.alpha,
            "tau": args.tau,
            "beta": args.beta,
            "sampling": args.sampling,
            "temperature": args.temperature if args.sampling else None,
            "top_p": args.top_p if args.sampling else None,
        },
        "results": []
    }
    
    # generate
    for batch_idx, batch in enumerate(eval_dataloader):
        prompts, question_ids, img_ids, input_ids, guidance_ids, images, guidance_images, attention_masks, guidance_attention_masks, sam_mask_batch = batch
        
        # Phase 4: Xử lý Mask với giá trị 1.0 (Foreground) và alpha (Background)
        # sam_mask_batch từ utils_dataset đang là tensor chứa 0 và 1.
        custom_sam_attention_mask = torch.where(sam_mask_batch == 1.0, 1.0, args.alpha)

        with torch.inference_mode():
            if args.decode_approach == 1:
                # Approach 1: Single Pass (Chỉ dùng Condition luồng với SAM Mask, không dùng CFG)
                output_ids = model.generate(
                    input_ids=guidance_ids, # Guidance_ids đã bao gồm cả Text Mồi và Prompt
                    pixel_values=guidance_images,
                    attention_mask=custom_sam_attention_mask,
                    do_sample=args.sampling,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    output_attentions=True
                )
                input_token_len = guidance_ids.shape[1]
            else:
                # Approach 2: Double Pass (Dùng CFG với Dynamic Gamma dựa trên Entropy)
                processor_obj = GuidanceLogits(tau=args.tau,
                                  beta=args.beta,
                                  guidance=guidance_ids,
                                  images=guidance_images,
                                  attention_mask=guidance_attention_masks, # Text mask cho Uncondition Pass
                                  sam_mask=custom_sam_attention_mask,      # Spatial Penalty Mask chứa 1.0 và alpha
                                  model=model,
                                  tokenizer=tokenizer,
                                  attention_catcher=catcher) # Truyền catcher vào
                                  
                output_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=images,
                    attention_mask=attention_masks, # Standard text mask cho Condition Pass
                    do_sample=args.sampling,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    output_attentions=True, # Bắt buộc để hook hứng được attention
                    logits_processor=LogitsProcessorList([
                        processor_obj
                    ])
                )
                processor_obj.clean_up() # Xóa hooks sau khi sinh xong để tránh memory leak
                input_token_len = input_ids.shape[1]

        # Batch decode the outputs
        decoded_outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)

        for i, output in enumerate(decoded_outputs):

            # Process each output
            output = output.strip()
            print(f"{question_ids[i]}: {output}")

            # Generate answer ID
            ans_id = shortuuid.uuid()
            final_output["results"].append({
                "question_id": question_ids[i],
                "image_id": img_ids[i],
                "prompt": prompts[i],
                "generated_text": output,
                "answer_id": ans_id
            })

    total_time = time.time() - start_time
    total_samples = len(final_output["results"])
    final_output["metadata"]["total_time_seconds"] = total_time
    final_output["metadata"]["total_samples"] = total_samples
    
    print(f"\n=========================================")
    print(f"Total Samples Processed: {total_samples}")
    print(f"Total Time Taken: {total_time:.2f} seconds")
    print(f"=========================================\n")

    json.dump(final_output, ans_file, indent=2)
    ans_file.close()
    print(f"Done! Saved answers to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image_folder", type=str,
                        default="./data/coco/val2014")
    parser.add_argument("--question_path", type=str,
                        default="./data/marine_qa/question")
    parser.add_argument("--question_file", type=str,
                        default="I02_mmc4_grey_ram_th0.68_detr_th0.95.json")
    parser.add_argument("--answer_path", type=str,
                        default="./data/marine_qa/answer")
    parser.add_argument("--answers_file", type=str, default=None)

    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=2.8, help="Tau parameter for dynamic gamma")
    parser.add_argument("--beta", type=float, default=3.0, help="Beta parameter for dynamic gamma")
    parser.add_argument("--alpha", type=float, default=0.7, help="Attention score ratio for background spatial tokens")
    parser.add_argument("--decode_approach", type=int, default=2, choices=[1, 2], help="1: Single Pass, 2: Double Pass")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--sampling", action="store_true")
    args = parser.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    eval_model(args)