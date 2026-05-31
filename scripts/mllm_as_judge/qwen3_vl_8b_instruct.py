import csv
import json
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import re

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import utils.prompts as prompt_lib

PROMPT_LIST = [prompt_lib.MLLM_AS_JUDGE_1, prompt_lib.MLLM_AS_JUDGE_2, prompt_lib.MLLM_AS_JUDGE_3, prompt_lib.MLLM_AS_JUDGE_4]
KEY_LIST = [
    "Illusion Fidelity", 
    "Overall Visual Quality", 
    "Semantic Consistency - Scene Consistency", 
    "Semantic Consistency - Illusion Shape Consistency"
]

BATCH_SIZE = 8
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

def create_blank_result(item_id):
    """Generate a structurally complete but empty record for use as a placeholder."""
    return {
        "id": str(item_id),
        "Illusion Fidelity": {"score": "", "reason": "Error or Missing"},
        "Overall Visual Quality": {"score": "", "reason": "Error or Missing"},
        "Semantic Consistency - Scene Consistency": {"score": "", "reason": "Error or Missing"},
        "Semantic Consistency - Illusion Shape Consistency": {"score": "", "reason": "Error or Missing"}
    }

def is_valid_result(res):
    """Check if a record is a valid (non-empty) result."""
    if not res:
        return False
    # Check if all scores are present
    for key in KEY_LIST:
        score = res.get(key, {}).get("score")
        if not score or not str(score).strip():
            return False
    return True

def process_batch(batch_items, model, processor, device, existing_records):
    """Process a batch of data and return parsed results."""
    valid_items = []
    images = []

    # Store the final merged results
    batch_results = []

    # 1. Preprocess data: skip samples missing images or prompts
    for item in batch_items:
        item_id = str(item.get('id'))
        bg_desc = item.get('Background Description')
        cam_desc = item.get('Camouflaged Object Description')
        image_path = os.path.join("data", "iaa", "images", f"iaa_{item_id}.png")

        # --- Get the existing partial record, or generate a blank placeholder if none exists ---
        current_record = existing_records.get(item_id, create_blank_result(item_id)).copy()

        if not os.path.exists(image_path) or not bg_desc or not cam_desc:
            print(f"[Skip] ID {item_id} data is incomplete or image not found.")
            batch_results.append(current_record)
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            valid_items.append(item)
            batch_results.append(current_record)

        except Exception as e:
            print(f"[Error] Unable to read image for ID {item_id}: {e}")
            batch_results.append(current_record)

    if not valid_items:
        return batch_results

    # 2. Run 4 loops, one question per loop
    for step_idx, prompt_template in enumerate(PROMPT_LIST):
        current_key = KEY_LIST[step_idx]

        # --- Filter out data missing scores for this specific dimension, only run the sub-metrics that haven't succeeded ---
        items_to_process = []
        images_to_process = []
        indices_in_batch = []

        for idx, item in enumerate(valid_items):
            item_id = str(item.get('id'))

            # Find the corresponding record index in batch_results
            record_idx = -1
            for i, r in enumerate(batch_results):
                if r['id'] == item_id:
                    record_idx = i
                    break

            score = batch_results[record_idx].get(current_key, {}).get("score")
            # Only add to the pending queue if this sub-metric currently has no valid score
            if not score or not str(score).strip():
                items_to_process.append(item)
                images_to_process.append(images[idx])
                indices_in_batch.append(record_idx)

        # If all images already have scores for this sub-metric, skip directly
        if not items_to_process:
            continue

        prompts = []

        for item in items_to_process:
            bg_desc = item.get('Background Description')
            cam_desc = item.get('Camouflaged Object Description')

            prompt_text = f"{prompt_template}\n\n**Task:**\nBackground Description: {bg_desc}\nCamouflaged Object Description: {cam_desc}\n\nEvaluate the image and output ONLY a valid JSON object:"
            prompts.append(prompt_text)

        try:
            messages_batch = []
            for prompt_text, img in zip(prompts, images_to_process):
                messages_batch.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ])

            # Use the processor to build the corresponding prompt template and tensors
            texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
            image_inputs, video_inputs = process_vision_info(messages_batch)

            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            # Generate with the required parameters
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=1024,
                top_p=0.9,
                repetition_penalty=1.0,
                temperature=0.1,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            generated_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        except Exception as e:
            print(f"[Model Inference Error] Dimension {current_key} batch processing failed: {e}")
            for i_batch in indices_in_batch:
                batch_results[i_batch][current_key] = {"score": "", "reason": "Model Exception"}
            continue

        # 3. Parse results
        for idx, (item, result_text) in enumerate(zip(items_to_process, generated_texts)):
            item_id = str(item.get('id'))
            i_batch = indices_in_batch[idx]  # Map back to the correct position in batch_results
            original_output = result_text.strip()

            # 1. Try to extract content after the task start to avoid capturing the Template from the first half of the Prompt
            if "**Task:**" in original_output:
                pure_output = original_output.split("**Task:**")[-1]
            else:
                pure_output = original_output

            # 2. Clean up Markdown code block markers (```json or ```)
            pure_output = re.sub(r'```json', '', pure_output)
            pure_output = re.sub(r'```', '', pure_output)

            # 3. Regex match: find the last pair of curly braces
            matches = re.findall(r'\{.*\}', pure_output, re.DOTALL)

            parsed_data = {"score": "", "reason": "Parse Failed"}

            if matches:
                clean_json_str = matches[-1].strip()
                try:
                    result_json = json.loads(clean_json_str)

                    if current_key in result_json:
                        val = result_json[current_key]
                    else:
                        val = result_json

                    score_val = val.get("score")
                    parsed_data["score"] = str(score_val) if score_val is not None else ""
                    parsed_data["reason"] = val.get("reason", "")
                except Exception as e:
                    print(f"Parse failed ID {item_id}: {e}")

            batch_results[i_batch][current_key] = parsed_data

    for res in batch_results:
        # Only print full success if all 4 metrics are complete
        if is_valid_result(res):
            print(f"[{MODEL_ID.split('/')[-1]}] Successfully evaluated ID: {res['id']}")

    return batch_results

def run_evaluation(dataset, output_file):
    print(f"\nStarting to load model {MODEL_ID} ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print("Model loaded, ready for evaluation...")

    # 1. Read historical records
    existing_records = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "id" in record:
                            existing_records[str(record["id"])] = record
                    except json.JSONDecodeError:
                        pass

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 2. Filter out unfinished tasks that need processing
    pending_items = []
    final_results_dict = existing_records.copy()

    for item in dataset:
        item_id = str(item.get('id'))
        if item_id in existing_records and is_valid_result(existing_records[item_id]):
            print(f"Skip already evaluated ID: {item_id}")
        else:
            pending_items.append(item)
            if item_id not in final_results_dict:
                final_results_dict[item_id] = create_blank_result(item_id)

    # 3. Batch processing by BATCH_SIZE
    print(f"Total {len(pending_items)} items to evaluate, Batch Size = {BATCH_SIZE}")

    for i in tqdm(range(0, len(pending_items), BATCH_SIZE), desc="Processing Batches"):
        batch = pending_items[i:i + BATCH_SIZE]
        batch_results = process_batch(batch, model, processor, device, final_results_dict)

        # Add new results to the dictionary
        for res in batch_results:
            final_results_dict[res["id"]] = res

        # 4. Overwrite the file after each batch to guard against mid-run crashes
        with open(output_file, "w", encoding="utf-8") as f:
            # Write in the original dataset ID order
            for original_item in dataset:
                o_id = str(original_item.get('id'))
                if o_id in final_results_dict:
                    f.write(json.dumps(final_results_dict[o_id], ensure_ascii=False) + "\n")

    print(f"Evaluation complete! Results saved in order to {output_file}")

def main():
    csv_path = os.path.join("data", "iaa", "iaa_dataset.csv")
    dataset = []

    if not os.path.exists(csv_path):
        print(f"[Fatal error] Dataset file not found: {csv_path}")
        return

    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)

    print(f"Successfully loaded {len(dataset)} raw records.")

    output_path = os.path.join("data", "mllm_as_judge", "qwen3_vl_8b_instruct.jsonl")
    run_evaluation(dataset, output_path)

if __name__ == "__main__":
    main()