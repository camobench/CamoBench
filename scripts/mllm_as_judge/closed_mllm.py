import asyncio
import csv
import json
import os
import base64
from dotenv import load_dotenv
from openai import AsyncOpenAI
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import utils.prompts as prompt_lib

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# Key list for the four dimensions
KEY_LIST = [
    "Illusion Fidelity",
    "Overall Visual Quality",
    "Semantic Consistency - Scene Consistency",
    "Semantic Consistency - Illusion Shape Consistency"
]

def encode_image(image_path):
    """Encode image to base64 format, compatible with Vision API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_blank_result(item_id):
    """Generate a structurally complete but empty record for placeholder use"""
    return {
        "id": str(item_id),
        "Illusion Fidelity": {"score": "", "reason": "Error or Missing"},
        "Overall Visual Quality": {"score": "", "reason": "Error or Missing"},
        "Semantic Consistency - Scene Consistency": {"score": "", "reason": "Error or Missing"},
        "Semantic Consistency - Illusion Shape Consistency": {"score": "", "reason": "Error or Missing"}
    }

def is_valid_result(res):
    """Check if a record is a valid (non-empty) result"""
    if not res:
        return False
    # Check if all 4 scores are present, to prevent incomplete data from mid-run interruption
    for key in KEY_LIST:
        score = res.get(key, {}).get("score")
        if not score or not str(score).strip():
            return False
    return True

async def process_item(item, model_name, semaphore, existing_result=None):
    """Process a single item: call API 4 times to evaluate 4 dimensions, then merge results"""
    item_id = str(item.get('id'))

    # --- If this ID already has a fully valid result, return it directly without calling API ---
    if existing_result and is_valid_result(existing_result):
        print(f"[{model_name}] ⏭️ Skip already evaluated ID: {item_id}")
        return existing_result

    async with semaphore:  # Control concurrency level
        bg_desc = item.get('Background Description')
        cam_desc = item.get('Camouflaged Object Description')
        
        # Assemble and validate
        image_path = os.path.join("data", "iaa", "images", f"iaa_{item_id}.png")

        if not os.path.exists(image_path):
            print(f"[Skip] Image file not found for ID {item_id}: {image_path}")
            return create_blank_result(item_id)

        if not bg_desc or not cam_desc:
            print(f"[Skip] ID {item_id} is missing prompt data.")
            return create_blank_result(item_id)

        # Try to read and encode the image
        try:
            base64_image = encode_image(image_path)
        except Exception as e:
            print(f"[Error] Unable to read image for ID {item_id}: {e}")
            return create_blank_result(item_id)

        # Dynamically get four prompt templates
        prompt_templates = [
            prompt_lib.MLLM_AS_JUDGE_1,
            prompt_lib.MLLM_AS_JUDGE_2,
            prompt_lib.MLLM_AS_JUDGE_3,
            prompt_lib.MLLM_AS_JUDGE_4
        ]

        final_output = {"id": item_id}
        
        # ================= Loop 4 times, evaluate one dimension per iteration =================
        for prompt_template, current_key in zip(prompt_templates, KEY_LIST):
            # Assemble API request content for a single evaluation
            messages = [
                {"role": "user", "content": [
                    {
                        "type": "text", 
                        "text": f"{prompt_template}\n\n---\n**Now, please evaluate the following image based on the rules above:**\nBackground Description: {bg_desc}\nCamouflaged Object Description: {cam_desc}\n\nOutput ONLY a valid JSON object."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]}
            ]

            try:
                # Call LLM API
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2
                )
                
                result_text = response.choices[0].message.content

                # Clean up Markdown markers
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                elif result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                result_text = result_text.strip()

                # Parse the JSON for this iteration
                result_json = json.loads(result_text)

                # Extract current dimension's result into final_output, and force numbers to strings
                if current_key in result_json:
                    val = result_json[current_key]
                    if isinstance(val, dict) and "score" in val:
                        val["score"] = str(val["score"])
                    final_output[current_key] = val
                else:
                    # Handle the case where model outputs {"score": "X", "reason": "Y"}
                    if "score" in result_json:
                        result_json["score"] = str(result_json["score"])
                        final_output[current_key] = result_json
                    else:
                        final_output[current_key] = {"score": "", "reason": "Parse Failed: Missing Key"}

            except Exception as e:
                print(f"[{model_name}] ❌ Error evaluating ID {item_id} dimension [{current_key}]: {e}")
                final_output[current_key] = {"score": "", "reason": f"API Error: {str(e)}"}

        if is_valid_result(final_output):
            print(f"[{model_name}] ✅ Successfully evaluated ID: {item_id}")
            
        return final_output

async def run_evaluation(model_name, dataset, output_file):
    """Schedule one model to process all data, supporting checkpoint resume"""
    print(f"\n🚀 Starting evaluation with {model_name}...")

    # 1. Read historical records (if file exists)
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
                        pass # Ignore corrupted lines

    # Create output directory (if not exists)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    semaphore = asyncio.Semaphore(50)
    
    # 2. Create all tasks and pass in existing records
    tasks = []
    for item in dataset:
        item_id = str(item.get('id'))
        existing_res = existing_records.get(item_id)
        tasks.append(process_item(item, model_name, semaphore, existing_res))

    # 3. Execute tasks
    # gather preserves task order, so results remain sorted 1-100
    results = await asyncio.gather(*tasks)

    # 4. Unified overwrite write
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            if res is not None:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"🎉 {model_name} evaluation complete! Results saved in order to {output_file}")

async def main():
    # 1. Load CSV data
    csv_path = os.path.join("data", "iaa", "iaa_dataset.csv")
    dataset = []

    if not os.path.exists(csv_path):
        print(f"[Fatal error] Dataset file not found: {csv_path}")
        return

    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)

    print(f"Successfully loaded {len(dataset)} records ready for testing.")

    # 2. Call two models in sequence
    await run_evaluation("gpt-5.2", dataset, os.path.join("data", "mllm_as_judge", "gpt5_2.jsonl"))
    await run_evaluation("gpt-5.4", dataset, os.path.join("data", "mllm_as_judge", "gpt5_4.jsonl"))
    await run_evaluation("gemini-3-pro-preview", dataset, os.path.join("data", "mllm_as_judge", "gemini3_pro_preview.jsonl"))

if __name__ == "__main__":
    asyncio.run(main())