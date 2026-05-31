import os
import json
import pandas as pd
from collections import defaultdict

# Configure file paths
SCORE_DIR = os.path.join('data', 'score')
DATASET_CSV = os.path.join('data', 'prompts', 'dataset.csv')
OUTPUT_DIR = os.path.join('data', 'score_analysis')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'rq3_combinations.csv')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("1. Reading dataset metadata...")
# Read dataset.csv and build id -> (Background, Object) mapping
meta_df = pd.read_csv(DATASET_CSV)
metadata = {}
for _, row in meta_df.iterrows():
    metadata[str(row['id'])] = (row['Background Category'], row['Camouflaged Object Super-category'])

# Define paradigm filename prefix mapping
paradigm_map = {
    'T2I_': 'T2I',
    'ConditionalGeneration_': 'Cond',
    'Editing_': 'Edit'
}

# Map full metric names to abbreviations
metric_map = {
    "Illusion Fidelity": "IF",
    "Overall Visual Quality": "VQ",
    "Semantic Consistency – Scene Consistency": "SC",
    "Semantic Consistency – Illusion Shape Consistency": "ISC"
}

# Nested dict to accumulate all matching scores: stats[bg][obj][paradigm][metric] = [score_list]
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

print("2. Parsing JSONL files and computing combined scores...")
# Iterate over all jsonl files in the score directory
for filename in os.listdir(SCORE_DIR):
    if not filename.endswith('.jsonl'):
        continue

    # Determine paradigm from filename prefix
    paradigm = None
    for prefix, p_name in paradigm_map.items():
        if filename.startswith(prefix):
            paradigm = p_name
            break

    if not paradigm:
        continue

    filepath = os.path.join(SCORE_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            item_id = str(data['id'])

            if item_id not in metadata:
                continue

            bg, obj = metadata[item_id]

            for raw_metric, short_metric in metric_map.items():
                if raw_metric in data:
                    scores_dict = data[raw_metric]
                    s1 = scores_dict.get('score1')
                    s2 = scores_dict.get('score2')

                    # Safe float conversion (filter out empty strings or format errors)
                    try:
                        s1 = float(s1) if s1 is not None and s1 != "" else None
                        s2 = float(s2) if s2 is not None and s2 != "" else None
                    except ValueError:
                        continue

                    # Core logic: average with score2 if present, otherwise use score1 directly
                    if s1 is not None:
                        if s2 is not None:
                            final_score = (s1 + s2) / 2.0
                        else:
                            final_score = s1

                        stats[bg][obj][paradigm][short_metric].append(final_score)

print("3. Computing the final 16x12 average matrix...")
# Fixed order for consistent table output
backgrounds = [
    "High-Frequency Texture",
    "Structured Texture",
    "Directional Texture",
    "Low-Frequency Smooth Texture"
]
objects = [
    "Textual Elements",
    "Geometry & Shapes",
    "Rigid Objects",
    "Non-Rigid Objects"
]
paradigms = ['T2I', 'Cond', 'Edit']
metrics = ['IF', 'VQ', 'SC', 'ISC']

rows = []
for bg in backgrounds:
    for obj in objects:
        row_dict = {'Background': bg, 'Hidden Object': obj}
        for p in paradigms:
            for m in metrics:
                scores = stats[bg][obj][p][m]
                col_name = f"{p}_{m}"
                if scores:
                    # Keep 3 decimal places for consistency with other tables
                    row_dict[col_name] = round(sum(scores) / len(scores), 3)
                else:
                    row_dict[col_name] = None  # Edge case: no data
        rows.append(row_dict)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"All done! Results exported to: {OUTPUT_FILE}")