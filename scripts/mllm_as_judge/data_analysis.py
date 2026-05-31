import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

# 1. Define file paths
iaa_file = "./data/iaa/iaa.jsonl"
mllm_dir = "./data/mllm_as_judge"

# 2. Define mapping between MLLM result keys and human annotation keys
metric_mapping = {
    "Illusion Fidelity": "Illusion Fidelity",
    "Overall Visual Quality": "Overall Visual Quality",
    "Semantic Consistency - Scene Consistency": "Semantic Consistency – Scene Consistency",
    "Semantic Consistency - Illusion Shape Consistency": "Semantic Consistency – Illusion Shape Consistency"
}

# 3. Read human annotation file and compute averages
human_scores = {}
with open(iaa_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        item_id = data['id']
        human_scores[item_id] = {}
        for human_metric in metric_mapping.values():
            if human_metric in data:
                s1 = float(data[human_metric]['score1'])
                s2 = float(data[human_metric]['score2'])
                s3 = float(data[human_metric]['score3'])
                human_scores[item_id][human_metric] = (s1 + s2 + s3) / 3.0

# 4. Print header
print(f"{'File_Name & Metric_Name':<85} {'PLCC':>10} {'SRCC':>10}")
print("-" * 107)

# 5. Iterate over MLLM output directory
if os.path.exists(mllm_dir):
    for filename in os.listdir(mllm_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(mllm_dir, filename)

            # Read all data from this MLLM file into an in-memory dict
            mllm_data = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    mllm_data[row['id']] = row

            # Compute correlations for each of the 4 metrics
            for mllm_metric, human_metric in metric_mapping.items():
                list_human = []
                list_auto = []

                # Iterate over all data IDs, matching human and MLLM scores
                for item_id, h_scores in human_scores.items():
                    if item_id in mllm_data and mllm_metric in mllm_data[item_id]:
                        h_score = h_scores[human_metric]
                        mllm_score_str = mllm_data[item_id][mllm_metric].get('score', '0')

                        try:
                            a_score = float(mllm_score_str)
                            list_human.append(h_score)
                            list_auto.append(a_score)
                        except ValueError:
                            # Filter out non-numeric strings from model hallucinations
                            continue

                # Build output row name: filename + metric name
                row_name = f"{filename} {mllm_metric}"

                # Compute PLCC and SRCC for this file/metric
                if len(list_human) > 1:
                    # If all values in the array are identical, output NaN directly
                    if len(np.unique(list_human)) == 1 or len(np.unique(list_auto)) == 1:
                        print(f"{row_name:<85} {'NaN':^12} {'NaN':^8}")
                    else:
                        plcc, _ = pearsonr(list_human, list_auto)
                        srcc, _ = spearmanr(list_human, list_auto)
                        print(f"{row_name:<85} {plcc:>10.4f} {srcc:>10.4f}")
                else:
                    print(f"{row_name:<55} {'NaN':^12} {'NaN':^8}")
else:
    print(f"Directory not found: {mllm_dir}")

'''
File_Name & Metric_Name                                                                     PLCC       SRCC
-----------------------------------------------------------------------------------------------------------
gemini3_pro_preview.jsonl Illusion Fidelity                                               0.3991     0.3751
gemini3_pro_preview.jsonl Overall Visual Quality                                          0.6835     0.6700
gemini3_pro_preview.jsonl Semantic Consistency - Scene Consistency                        0.6925     0.6992
gemini3_pro_preview.jsonl Semantic Consistency - Illusion Shape Consistency               0.4885     0.5025
gemini3_pro_preview_resize.jsonl Illusion Fidelity                                        0.5044     0.4698
gemini3_pro_preview_resize.jsonl Overall Visual Quality                                   0.7024     0.7011
gemini3_pro_preview_resize.jsonl Semantic Consistency - Scene Consistency                 0.7182     0.7256
gemini3_pro_preview_resize.jsonl Semantic Consistency - Illusion Shape Consistency        0.5319     0.5169
gpt5_2.jsonl Illusion Fidelity                                                            0.0109    -0.0406
gpt5_2.jsonl Overall Visual Quality                                                       0.5537     0.4910
gpt5_2.jsonl Semantic Consistency - Scene Consistency                                     0.5878     0.5951
gpt5_2.jsonl Semantic Consistency - Illusion Shape Consistency                            0.3610     0.3684
gpt5_2_resize.jsonl Illusion Fidelity                                                     0.1899     0.1489
gpt5_2_resize.jsonl Overall Visual Quality                                                0.5709     0.5445
gpt5_2_resize.jsonl Semantic Consistency - Scene Consistency                              0.6186     0.6145
gpt5_2_resize.jsonl Semantic Consistency - Illusion Shape Consistency                     0.4226     0.4225
instructblip_vicuna_7b.jsonl Illusion Fidelity                                            NaN        NaN
instructblip_vicuna_7b.jsonl Overall Visual Quality                                       0.2386     0.2128
instructblip_vicuna_7b.jsonl Semantic Consistency - Scene Consistency                     NaN        NaN
instructblip_vicuna_7b.jsonl Semantic Consistency - Illusion Shape Consistency            NaN        NaN
qwen3_vl_2b_instruct.jsonl Illusion Fidelity                                              0.1929     0.2287
qwen3_vl_2b_instruct.jsonl Overall Visual Quality                                         0.0811     0.1003
qwen3_vl_2b_instruct.jsonl Semantic Consistency - Scene Consistency                       0.1882     0.2057
qwen3_vl_2b_instruct.jsonl Semantic Consistency - Illusion Shape Consistency              0.1412     0.1566
qwen3_vl_2b_instruct_resize.jsonl Illusion Fidelity                                       0.3224     0.3452
qwen3_vl_2b_instruct_resize.jsonl Overall Visual Quality                                  0.0904     0.2381
qwen3_vl_2b_instruct_resize.jsonl Semantic Consistency - Scene Consistency                0.2034     0.2119
qwen3_vl_2b_instruct_resize.jsonl Semantic Consistency - Illusion Shape Consistency       0.2304     0.2521
qwen3_vl_8b_instruct.jsonl Illusion Fidelity                                             -0.2385    -0.2592
qwen3_vl_8b_instruct.jsonl Overall Visual Quality                                         0.3144     0.3027
qwen3_vl_8b_instruct.jsonl Semantic Consistency - Scene Consistency                       0.6613     0.6412
qwen3_vl_8b_instruct.jsonl Semantic Consistency - Illusion Shape Consistency              0.1495     0.1531
'''