import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

# 1. Define file paths
iaa_file = "./data/iaa/iaa.jsonl"
score_file = "./data/existing_automation_metrics/score.jsonl"

# 2. Define mapping between automated metrics and human metrics
mapping = {
    "Illusion Fidelity - DeltaVQAScore": "Illusion Fidelity",
    "Overall Visual Quality - NIQE": "Overall Visual Quality",
    "Overall Visual Quality - MUSIQ": "Overall Visual Quality",
    "Overall Visual Quality - LAION-Aes": "Overall Visual Quality",
    "Semantic Consistency – Scene Consistency - CLIP": "Semantic Consistency – Scene Consistency",
    "Semantic Consistency – Scene Consistency - BLIP": "Semantic Consistency – Scene Consistency",
    "Semantic Consistency – Scene Consistency - VQAScore": "Semantic Consistency – Scene Consistency",
    "Semantic Consistency – Illusion Shape Consistency - CLIP": "Semantic Consistency – Illusion Shape Consistency",
    "Semantic Consistency – Illusion Shape Consistency - BLIP": "Semantic Consistency – Illusion Shape Consistency",
    "Semantic Consistency – Illusion Shape Consistency - VQAScore": "Semantic Consistency – Illusion Shape Consistency"
}

# 3. Negate metrics where lower is better
lower_is_better_metrics = ["Overall Visual Quality - NIQE"]

# 4. Read human annotation file and compute averages
human_scores = {}
with open(iaa_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        item_id = data['id']
        human_scores[item_id] = {}
        # Iterate over 4 human metrics and compute average scores
        for metric in ["Illusion Fidelity", "Overall Visual Quality", "Semantic Consistency – Scene Consistency", "Semantic Consistency – Illusion Shape Consistency"]:
            if metric in data:
                s1 = float(data[metric]['score1'])
                s2 = float(data[metric]['score2'])
                s3 = float(data[metric]['score3'])
                avg_score = (s1 + s2 + s3) / 3.0
                human_scores[item_id][metric] = avg_score

# 5. Read automated scoring file
auto_scores = {}
with open(score_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        item_id = data['id']
        auto_scores[item_id] = data

# 6. Compute correlations and output as required
output_order = [
    "Illusion Fidelity - DeltaVQAScore",
    "Overall Visual Quality - NIQE",
    "Overall Visual Quality - MUSIQ",
    "Overall Visual Quality - LAION-Aes",
    "Semantic Consistency – Scene Consistency - CLIP",
    "Semantic Consistency – Scene Consistency - BLIP",
    "Semantic Consistency – Scene Consistency - VQAScore",
    "Semantic Consistency – Illusion Shape Consistency - CLIP",
    "Semantic Consistency – Illusion Shape Consistency - BLIP",
    "Semantic Consistency – Illusion Shape Consistency - VQAScore"
]

print("-" * 78)
print(f"{'Metric_Name':<60} {'PLCC':>8} {'SRCC':>8}")
print("-" * 78)

for auto_metric in output_order:
    human_metric = mapping[auto_metric]
    
    list_human = []
    list_auto = []
    
    for item_id in human_scores:
        if item_id in auto_scores and auto_metric in auto_scores[item_id]:
            h_score = human_scores[item_id][human_metric]
            a_score = float(auto_scores[item_id][auto_metric])
            
            # If this metric is lower-is-better, multiply by -1
            if auto_metric in lower_is_better_metrics:
                a_score = -a_score
                
            list_human.append(h_score)
            list_auto.append(a_score)
            
    # Compute PLCC and SRCC
    if len(list_human) > 1:
        plcc, _ = pearsonr(list_human, list_auto)
        srcc, _ = spearmanr(list_human, list_auto)
        # Format output to 4 decimal places
        print(f"{auto_metric:<60} {plcc:>8.4f} {srcc:>8.4f}")
    else:
        print(f"{auto_metric:<60} {'N/A':>8} {'N/A':>8}")

'''
------------------------------------------------------------------------------
Metric_Name                                                      PLCC     SRCC
------------------------------------------------------------------------------
Illusion Fidelity - DeltaVQAScore                              0.2730   0.2425
Overall Visual Quality - NIQE                                  0.4529   0.4282
Overall Visual Quality - MUSIQ                                 0.1927   0.1851
Overall Visual Quality - LAION-Aes                             0.4347   0.3733
Semantic Consistency – Scene Consistency - CLIP                0.6667   0.6548
Semantic Consistency – Scene Consistency - BLIP                0.5886   0.6003
Semantic Consistency – Scene Consistency - VQAScore            0.6746   0.6437
Semantic Consistency – Illusion Shape Consistency - CLIP       0.1107   0.0719
Semantic Consistency – Illusion Shape Consistency - BLIP       0.2173   0.1356
Semantic Consistency – Illusion Shape Consistency - VQAScore   0.2027   0.2135
'''