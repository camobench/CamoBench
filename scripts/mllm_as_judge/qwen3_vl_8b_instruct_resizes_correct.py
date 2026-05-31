import csv
import json
import os
import sys
import torch
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import utils.prompts as prompt_lib

# Only evaluate the two dimensions relevant to MDS
KEY_LIST = [
    "Illusion Fidelity",
    "Semantic Consistency - Illusion Shape Consistency"
]

# Corresponding two-turn prompts
PROMPTS_TURN_1 = [prompt_lib.MLLM_AS_JUDGE_1, prompt_lib.MLLM_AS_JUDGE_4]
PROMPTS_TURN_2 = [prompt_lib.MLLM_AS_JUDGE_1_RESIZES_CORRECT, prompt_lib.MLLM_AS_JUDGE_4_RESIZES_CORRECT]

BATCH_SIZE = 4  # 32G VRAM handles Batch Size 4 with ease
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

def create_blank_result(item_id):
    """Generate a structurally complete but empty record for use as a placeholder."""
    return {
        "id": str(item_id),
        "Illusion Fidelity": {"score": "", "reason": "Error or Missing"},
        "Semantic Consistency - Illusion Shape Consistency": {"score": "", "reason": "Error or Missing"}
    }

def is_valid_result(res):
    """Check if a record is a valid (non-empty) result."""
    if not res:
        return False
    for key in KEY_LIST:
        score = res.get(key, {}).get("score")
        if not score or not str(score).strip():
            return False
    return True

def process_batch_mds(batch_items, model, processor, device, existing_records):
    """Process a batch of data and return parsed results (two-turn conversation MDS logic)."""
    valid_items = []
    images_original = []
    images_128 = []
    images_64 = []
    images_32 = []

    batch_results = []

    # 1. Preprocess data: load original images and generate resized versions
    for item in batch_items:
        item_id = str(item.get('id'))
        bg_desc = item.get('Background Description')
        cam_desc = item.get('Camouflaged Object Description')
        image_path = os.path.join("data", "iaa", "images", f"iaa_{item_id}.png")

        current_record = existing_records.get(item_id, create_blank_result(item_id)).copy()

        if not os.path.exists(image_path) or not bg_desc or not cam_desc:
            print(f"[Skip] ID {item_id} data is incomplete or image not found.")
            batch_results.append(current_record)
            continue

        try:
            img_orig = Image.open(image_path).convert("RGB")
            # Generate resized versions
            img_128 = img_orig.resize((128, 128))
            img_64 = img_orig.resize((64, 64))
            img_32 = img_orig.resize((32, 32))

            images_original.append(img_orig)
            images_128.append(img_128)
            images_64.append(img_64)
            images_32.append(img_32)
            valid_items.append(item)
            batch_results.append(current_record)

        except Exception as e:
            print(f"[Error] Unable to read image for ID {item_id}: {e}")
            batch_results.append(current_record)

    if not valid_items:
        return batch_results

    # 2. Evaluate IF and ISC separately
    for step_idx, current_key in enumerate(KEY_LIST):
        p_turn1 = PROMPTS_TURN_1[step_idx]
        p_turn2 = PROMPTS_TURN_2[step_idx]

        # Filter out data missing the current metric
        items_to_process = []
        indices_in_batch = []

        for idx, item in enumerate(valid_items):
            item_id = str(item.get('id'))
            record_idx = next(i for i, r in enumerate(batch_results) if r['id'] == item_id)

            score = batch_results[record_idx].get(current_key, {}).get("score")
            if not score or not str(score).strip():
                items_to_process.append((idx, item))  # Save index in valid_items for image retrieval
                indices_in_batch.append(record_idx)

        if not items_to_process:
            continue

        # ================= Turn 1 (original image only) =================
        messages_batch_turn1 = []
        for valid_idx, item in items_to_process:
            bg_desc = item.get('Background Description')
            cam_desc = item.get('Camouflaged Object Description')

            prompt_text_1 = f"{p_turn1}\n\n---\n**Now, please evaluate the following images based on the rules above:**\nBackground Description: {bg_desc}\nCamouflaged Object Description: {cam_desc}\n\nOutput ONLY a valid JSON object."

            messages_batch_turn1.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images_original[valid_idx]},
                        {"type": "text", "text": prompt_text_1}
                    ]
                }
            ])

        try:
            texts_1 = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch_turn1]
            image_inputs_1, video_inputs_1 = process_vision_info(messages_batch_turn1)

            inputs_1 = processor(text=texts_1, images=image_inputs_1, videos=video_inputs_1, padding=True, return_tensors="pt").to(device)

            outputs_1 = model.generate(**inputs_1, do_sample=True, max_new_tokens=1024, temperature=0.2)
            generated_ids_1 = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_1.input_ids, outputs_1)]
            responses_turn1 = processor.batch_decode(generated_ids_1, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        except Exception as e:
            print(f"[Model Inference Error] Turn 1 {current_key} failed: {e}")
            for i_batch in indices_in_batch:
                batch_results[i_batch][current_key] = {"score": "", "reason": "Model Exception Turn 1"}
            continue

        # ================= Turn 2 (history + 3 resized images) =================
        messages_batch_turn2 = []
        for (valid_idx, item), response_1 in zip(items_to_process, responses_turn1):
            # Build complete conversation including turn-1 history
            msg_turn2 = messages_batch_turn1[valid_idx].copy()  # User Turn 1
            msg_turn2.append({"role": "assistant", "content": response_1})  # Assistant Turn 1
            msg_turn2.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": p_turn2},
                    {"type": "image", "image": images_128[valid_idx]},
                    {"type": "image", "image": images_64[valid_idx]},
                    {"type": "image", "image": images_32[valid_idx]}
                ]
            })
            messages_batch_turn2.append(msg_turn2)

        try:
            texts_2 = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch_turn2]
            image_inputs_2, video_inputs_2 = process_vision_info(messages_batch_turn2)

            inputs_2 = processor(text=texts_2, images=image_inputs_2, videos=video_inputs_2, padding=True, return_tensors="pt").to(device)

            outputs_2 = model.generate(**inputs_2, do_sample=True, max_new_tokens=1024, temperature=0.2)
            generated_ids_2 = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_2.input_ids, outputs_2)]
            responses_turn2 = processor.batch_decode(generated_ids_2, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        except Exception as e:
            print(f"[Model Inference Error] Turn 2 {current_key} failed: {e}")
            for i_batch in indices_in_batch:
                batch_results[i_batch][current_key] = {"score": "", "reason": "Model Exception Turn 2"}
            continue

        # ================= Parse turn-2 results =================
        for idx, ((valid_idx, item), result_text) in enumerate(zip(items_to_process, responses_turn2)):
            item_id = str(item.get('id'))
            i_batch = indices_in_batch[idx]

            pure_output = re.sub(r'```json', '', result_text.strip())
            pure_output = re.sub(r'```', '', pure_output)
            matches = re.findall(r'\{.*\}', pure_output, re.DOTALL)

            parsed_data = {"score": "", "reason": "Parse Failed"}
            if matches:
                clean_json_str = matches[-1].strip()
                try:
                    result_json = json.loads(clean_json_str)
                    val = result_json.get(current_key, result_json)
                    score_val = val.get("score")
                    parsed_data["score"] = str(score_val) if score_val is not None else ""
                    parsed_data["reason"] = val.get("reason", "")
                except Exception as e:
                    print(f"Parse failed ID {item_id}: {e}")

            batch_results[i_batch][current_key] = parsed_data

    for res in batch_results:
        if is_valid_result(res):
            print(f"[{MODEL_ID.split('/')[-1]}] MDS successfully evaluated ID: {res['id']}")

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
    print("Model loaded, ready for MDS evaluation...")

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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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

    print(f"Total {len(pending_items)} items to evaluate, Batch Size = {BATCH_SIZE}")

    for i in tqdm(range(0, len(pending_items), BATCH_SIZE), desc="Processing Batches"):
        batch = pending_items[i:i + BATCH_SIZE]
        batch_results = process_batch_mds(batch, model, processor, device, final_results_dict)

        for res in batch_results:
            final_results_dict[res["id"]] = res

        with open(output_file, "w", encoding="utf-8") as f:
            for original_item in dataset:
                o_id = str(original_item.get('id'))
                if o_id in final_results_dict:
                    f.write(json.dumps(final_results_dict[o_id], ensure_ascii=False) + "\n")

    print(f"MDS evaluation complete! Results saved to {output_file}")

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

    # Save with resizes_correct suffix to indicate MDS strategy results
    output_path = os.path.join("data", "mllm_as_judge", "qwen3_vl_8b_instruct_resizes_correct.jsonl")
    run_evaluation(dataset, output_path)

if __name__ == "__main__":
    main()