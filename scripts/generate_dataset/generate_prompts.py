import sys
import os
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import utils.prompts as prompt_lib

# === 1. Configuration ===
DATA_DIR = BASE_DIR / "data" / "processed"
INPUT_BACKGROUND_FILE = DATA_DIR / "background_set.csv"
INPUT_OBJECT_FILE = DATA_DIR / "camouflaged_object_set.csv"
OUTPUT_FILE = DATA_DIR / "dataset.csv"

# Background category names
BACKGROUND_CATEGORIES = [
    "High-Frequency Texture",
    "Structured Texture",
    "Directional Texture",
    "Low-Frequency Smooth Texture"
]

# API configuration
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.7
CONCURRENCY_LIMIT = 64  # Number of concurrent requests

# === 2. Initialization ===
load_dotenv()
aclient = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

async def generate_single_prompt(bg_desc, obj_desc, semaphore):
    """
    Call LLM to generate a prompt, using a semaphore to control concurrency.
    Returns an empty string on failure.
    """
    async with semaphore:
        user_content = f"Background Scene: {bg_desc}\nHidden Concept: {obj_desc}"
        try:
            response = await aclient.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt_lib.DATASET_CONSTRUCTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content:
                # Strip ```json, ```, and surrounding whitespace
                content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            return result.get("prompt", "")
        except Exception as e:
            print(f"[Error] Generating prompt for '{obj_desc}' + '{bg_desc[:15]}...': {e}")
            return ""

async def main():
    # === 3. Read data ===
    print("Reading CSV files...")
    try:
        bg_df = pd.read_csv(INPUT_BACKGROUND_FILE)
        obj_df = pd.read_csv(INPUT_OBJECT_FILE)
    except FileNotFoundError as e:
        print(f"Error: file not found {e.filename}")
        return

    # Preprocess background data: build {Category: [desc_list]} dictionary
    bg_map = {}
    for cat in BACKGROUND_CATEGORIES:
        descs = bg_df[bg_df['Background Category'] == cat]['Description'].tolist()
        if len(descs) == 0:
            print(f"Warning: background category '{cat}' is empty!")
        bg_map[cat] = descs

    # === 4. Build task list ===
    tasks_data = []  # Store all pending row data (without Prompt)

    print("Building task queue...")
    # Iterate over each camouflaged object
    # Note: obj_df index starts at 0, corresponding to object 1, object 2...
    for obj_idx, obj_row in obj_df.iterrows():
        
        # Read object info
        obj_super = obj_row['Super-category']
        obj_sub = obj_row['Sub-category']
        obj_desc = obj_row['Description']
        # Read Reference Image column from CSV
        obj_ref_img = obj_row.get('Reference Image', '') 

        # The logic is:
        # Object 1 (idx=0) -> matches background list index 0
        # Object 2 (idx=1) -> matches background list index 1
        # Object 33 (idx=32) -> matches background list index 0 (32 % 32 = 0)
        bg_match_index = obj_idx % 32

        # Each object pairs with all 4 background categories
        for bg_cat in BACKGROUND_CATEGORIES:
            bg_list = bg_map.get(bg_cat, [])

            # Ensure enough background descriptions exist in this category
            if bg_list and len(bg_list) > bg_match_index:
                bg_desc = bg_list[bg_match_index]
                
                # Build the data structure for this row (staging)
                row_data = {
                    "Background Category": bg_cat,
                    "Background Description": bg_desc,
                    "Camouflaged Object Super-category": obj_super,
                    "Camouflaged Object Sub-category": obj_sub,
                    "Camouflaged Object Description": obj_desc,
                    "Camouflaged Object Reference Image": obj_ref_img,
                    # "Prompt": to be generated
                }
                tasks_data.append(row_data)
            else:
                print(f"Error: category {bg_cat} missing entry {bg_match_index}, skipping combination.")

    total_tasks = len(tasks_data)
    print(f"Generated {total_tasks} combined tasks. Starting concurrent LLM calls...")

    # === 5. Concurrent execution and batch saving ===
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    # Initialize CSV file (write header if file doesn't exist; overwrite if it does)
    # Column order: id, Bg Cat, Bg Desc, Obj Super, Obj Sub, Obj Desc, Ref Img, Prompt
    columns = [
        "id", 
        "Background Category", 
        "Background Description", 
        "Camouflaged Object Super-category", 
        "Camouflaged Object Sub-category", 
        "Camouflaged Object Description", 
        "Camouflaged Object Reference Image", 
        "Prompt"
    ]
    
    # Create an empty DataFrame with headers to clear any old file
    pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Process in batches
    batch_size = CONCURRENCY_LIMIT  # Process 16 at a time
    global_id_counter = 1

    for i in range(0, total_tasks, batch_size):
        batch_rows = tasks_data[i : i + batch_size]

        # Create async tasks
        prompt_tasks = []
        for row in batch_rows:
            prompt_tasks.append(
                generate_single_prompt(row["Background Description"], row["Camouflaged Object Description"], semaphore)
            )

        # Wait for this batch to finish
        print(f"Processing batch {i//batch_size + 1}/{(total_tasks + batch_size - 1)//batch_size} ...")
        batch_prompts = await asyncio.gather(*prompt_tasks)

        # Assemble results
        final_batch_data = []
        for row, prompt_text in zip(batch_rows, batch_prompts):
            # Build the complete row
            complete_row = {
                "id": global_id_counter,
                "Background Category": row["Background Category"],
                "Background Description": row["Background Description"],
                "Camouflaged Object Super-category": row["Camouflaged Object Super-category"],
                "Camouflaged Object Sub-category": row["Camouflaged Object Sub-category"],
                "Camouflaged Object Description": row["Camouflaged Object Description"],
                "Camouflaged Object Reference Image": row["Camouflaged Object Reference Image"],
                "Prompt": prompt_text
            }
            final_batch_data.append(complete_row)
            global_id_counter += 1

        # Immediately append to CSV (mode='a', header=False)
        df_batch = pd.DataFrame(final_batch_data)
        # Enforce column order to ensure clean output
        df_batch = df_batch[columns]
        df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")

        print(f"Batch saved. Progress: {min(i + batch_size, total_tasks)}/{total_tasks}")

    print(f"\nAll done! File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())