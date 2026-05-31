import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix

# 1. Define folder paths
iaa_file = "./data/iaa/iaa.jsonl"
mllm_dir = "./data/mllm_as_judge"
cm_output_dir = os.path.join(mllm_dir, "confusion_matrix")

# Ensure output directory exists
os.makedirs(cm_output_dir, exist_ok=True)

# 2. Define mapping relationships and valid score ranges for each metric (used for fixing the confusion matrix coordinate system)
metric_mapping = {
    "Illusion Fidelity": "Illusion Fidelity",
    "Overall Visual Quality": "Overall Visual Quality",
    "Semantic Consistency - Scene Consistency": "Semantic Consistency – Scene Consistency",
    "Semantic Consistency - Illusion Shape Consistency": "Semantic Consistency – Illusion Shape Consistency"
}

score_ranges = {
    "Illusion Fidelity": [1, 2, 3, 4, 5],
    "Overall Visual Quality": [1, 2, 3, 4, 5],
    "Semantic Consistency - Scene Consistency": [0, 1],
    "Semantic Consistency - Illusion Shape Consistency": [1, 2, 3]
}

# 3. Read human scoring file (save both mean and raw 3 scores for confusion matrix)
human_scores = {}
with open(iaa_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        item_id = data['id']
        human_scores[item_id] = {}
        for mllm_metric, human_metric in metric_mapping.items():
            if human_metric in data:
                s1 = float(data[human_metric]['score1'])
                s2 = float(data[human_metric]['score2'])
                s3 = float(data[human_metric]['score3'])
                # mean used for PLCC/SRCC, raw used for confusion matrix
                human_scores[item_id][mllm_metric] = {
                    'mean': (s1 + s2 + s3) / 3.0,
                    'raw': [int(s1), int(s2), int(s3)]
                }

# 4. Print header
print(f"{'File_Name & Metric_Name':<85} {'PLCC':>10} {'SRCC':>10}")
print("-" * 107)

# 5. Iterate over MLLM folder
if os.path.exists(mllm_dir):
    for filename in os.listdir(mllm_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(mllm_dir, filename)

            # First read all data from this MLLM file into a dictionary in memory
            mllm_data = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    mllm_data[row['id']] = row
            
            # Store data for drawing confusion matrix of current file
            cm_data = {m: {'y_true': [], 'y_pred': []} for m in metric_mapping.keys()}

            # Loop through the 4 metrics separately
            for mllm_metric in metric_mapping.keys():
                list_human_mean = []
                list_auto_mean = []

                # Iterate over all data IDs, match human and MLLM scores
                for item_id, h_scores_dict in human_scores.items():
                    if item_id in mllm_data and mllm_metric in mllm_data[item_id] and mllm_metric in h_scores_dict:
                        h_score_mean = h_scores_dict[mllm_metric]['mean']
                        h_scores_raw = h_scores_dict[mllm_metric]['raw']
                        
                        mllm_score_str = mllm_data[item_id][mllm_metric].get('score', '0')
                        
                        try:
                            a_score_float = float(mllm_score_str)
                            a_score_int = int(round(a_score_float)) # Convert to integer for confusion matrix

                            # Collect PLCC/SRCC data (1-to-1)
                            list_human_mean.append(h_score_mean)
                            list_auto_mean.append(a_score_float)

                            # Collect confusion matrix data (1-to-3, sample size x3)
                            # If the model outputs an out-of-range hallucination score, keep it anyway, but it may be ignored or classified in the axes
                            for h_raw in h_scores_raw:
                                cm_data[mllm_metric]['y_true'].append(h_raw)
                                cm_data[mllm_metric]['y_pred'].append(a_score_int)
                                
                        except ValueError:
                            # Filter out non-numeric strings from model hallucination output
                            continue

                # Concatenate output row name: file name + metric name
                row_name = f"{filename} {mllm_metric}"

                # Compute PLCC and SRCC for this metric under this file
                if len(list_human_mean) > 1:
                    if len(np.unique(list_human_mean)) == 1 or len(np.unique(list_auto_mean)) == 1:
                        print(f"{row_name:<85} {'NaN':^12} {'NaN':^8}")
                    else:
                        plcc, _ = pearsonr(list_human_mean, list_auto_mean)
                        srcc, _ = spearmanr(list_human_mean, list_auto_mean)
                        print(f"{row_name:<85} {plcc:>10.4f} {srcc:>10.4f}")
                else:
                    # No valid data exists
                    pass

            # ================= Plotting module =================
            # Filter metrics with valid data
            valid_metrics = [m for m in metric_mapping.keys() if len(cm_data[m]['y_true']) > 0]
            num_metrics = len(valid_metrics)
            
            if num_metrics > 0:
                # Dynamic layout: if 3-4 metrics, use 2 columns; if 1-2, use rows equal to metric count
                cols = 2 if num_metrics >= 3 else 1
                rows = (num_metrics + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
                # Ensure axes is a flat list for easy iteration
                axes_flat = np.atleast_1d(axes).flatten()
                
                for idx, m in enumerate(valid_metrics):
                    ax = axes_flat[idx]
                    y_true = cm_data[m]['y_true']
                    y_pred = cm_data[m]['y_pred']
                    labels = score_ranges[m]
                    
                    # Use sklearn to compute confusion matrix, specify labels to ensure axes show full range even if some scores don't appear
                    cm = confusion_matrix(y_true, y_pred, labels=labels)

                    # Draw heatmap
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=labels, yticklabels=labels)
                    ax.set_title(f"{m}\n(Total Samples: {len(y_true)})", fontsize=12)
                    ax.set_xlabel('MLLM Predicted Score', fontsize=10)
                    ax.set_ylabel('Human Ground Truth', fontsize=10)
                
                # Hide extra blank subplots (e.g., when there are 3 metrics, 1 will be blank)
                for i in range(num_metrics, len(axes_flat)):
                    fig.delaxes(axes_flat[i])

                plt.tight_layout()

                # Save image with jsonl file name (remove .jsonl suffix)
                base_name = filename.replace('.jsonl', '')
                save_path = os.path.join(cm_output_dir, f"{base_name}_confusion_matrix.png")
                plt.savefig(save_path, dpi=300)
                plt.close(fig)
else:
    print(f"Folder not found: {mllm_dir}")