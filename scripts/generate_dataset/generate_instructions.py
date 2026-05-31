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

# === 1. Configuration Section ===
DATA_DIR = BASE_DIR / "data" / "processed"
INPUT_BACKGROUND_FILE = DATA_DIR / "background_set.csv"
INPUT_DATASET_FILE = DATA_DIR / "dataset.csv" 
OUTPUT_FILE = DATA_DIR / "dataset.csv"

# API Configuration
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.7
CONCURRENCY_LIMIT = 64 

# === 2. Initialization ===
load_dotenv()
aclient = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

async def generate_instruction(bg_desc, bg_cat, obj_desc, semaphore):
    async with semaphore:
        user_content = (
            f"Source Image Description: {bg_desc}\n"
            f"Texture Type: {bg_cat}\n"
            f"Hidden Concept: {obj_desc}"
        )
        try:
            response = await aclient.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt_lib.IMAGE_EDITING_MODEL_DATASET_CONSTRUCTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content:
                content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            return result.get("instruction", "") 
        except Exception as e:
            print(f"[Error] Generating instruction for '{obj_desc[:15]}...': {e}")
            return ""

async def main():
    # === 3. Read Data ===
    print("Reading CSV files...")
    try:
        bg_set_df = pd.read_csv(INPUT_BACKGROUND_FILE)
        dataset_df = pd.read_csv(INPUT_DATASET_FILE)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
        return

    # Build background image mapping dictionary
    bg_ref_map = dict(zip(bg_set_df['Description'], bg_set_df['Reference Image']))

    # === 4. Build Task List ===
    tasks_data = []
    print("Matching data and building task queue...")

    for _, row in dataset_df.iterrows():
        bg_desc = row['Background Description']

        # Get background reference image
        bg_ref_img = bg_ref_map.get(bg_desc, "")

        # Build row base data, preserving original Prompt
        row_data = {
            "Background Category": row['Background Category'],
            "Background Description": bg_desc,
            "Background Reference Image": bg_ref_img,
            "Camouflaged Object Super-category": row['Camouflaged Object Super-category'],
            "Camouflaged Object Sub-category": row['Camouflaged Object Sub-category'],
            "Camouflaged Object Description": row['Camouflaged Object Description'],
            "Camouflaged Object Reference Image": row['Camouflaged Object Reference Image'],
            "Prompt": row['Prompt']
        }
        tasks_data.append(row_data)

    total_tasks = len(tasks_data)
    print(f"Total {total_tasks} tasks. Starting concurrent Instruction generation...")

    # === 5. Concurrent Execution and Batch Saving ===
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # Final column order: original columns + newly inserted Bg Ref Img + Instruction at the end
    columns = [
        "id",
        "Background Category",
        "Background Description",
        "Background Reference Image",
        "Camouflaged Object Super-category",
        "Camouflaged Object Sub-category",
        "Camouflaged Object Description",
        "Camouflaged Object Reference Image",
        "Prompt",
        "Instruction" # New column
    ]

    # Initialize output file
    pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    batch_size = CONCURRENCY_LIMIT
    global_id_counter = 1

    for i in range(0, total_tasks, batch_size):
        batch_rows = tasks_data[i : i + batch_size]
        
        # Async generate new content (Instruction)
        instruction_tasks = []
        for row in batch_rows:
            instruction_tasks.append(
                generate_instruction(
                    row["Background Description"],
                    row["Background Category"],
                    row["Camouflaged Object Description"],
                    semaphore
                )
            )

        print(f"Processing batch {i//batch_size + 1}/{(total_tasks + batch_size - 1)//batch_size} ...")
        batch_instructions = await asyncio.gather(*instruction_tasks)

        # Assemble data
        final_batch_data = []
        for row, inst_text in zip(batch_rows, batch_instructions):
            complete_row = row.copy()
            complete_row["id"] = global_id_counter
            complete_row["Instruction"] = inst_text # Fill new column
            final_batch_data.append(complete_row)
            global_id_counter += 1

        # Write to CSV
        df_batch = pd.DataFrame(final_batch_data)
        df_batch = df_batch[columns]
        df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")

        print(f"Batch saved. Current progress: {min(i + batch_size, total_tasks)}/{total_tasks}")

    print(f"\nAll done! Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())