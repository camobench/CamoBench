import os
import glob
import json
import csv

def analyze_scores():
    # Define input and output paths
    input_dir = os.path.join('data', 'score')
    output_dir = os.path.join('data', 'score_analysis')
    output_csv = os.path.join(output_dir, 'all.csv')

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the 4 metrics to be analyzed
    metrics = [
        "Illusion Fidelity",
        "Overall Visual Quality",
        "Semantic Consistency – Scene Consistency",
        "Semantic Consistency – Illusion Shape Consistency"
    ]

    # Prepare to write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        
        # Write header: filename + 4 metrics
        header = ['Filename'] + metrics
        writer.writerow(header)

        # Read all jsonl files
        search_pattern = os.path.join(input_dir, '*.jsonl')
        for filepath in glob.glob(search_pattern):
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(filepath))[0]
            
            # Initialize metric sums and counters for current file
            metric_sums = {metric: 0.0 for metric in metrics}
            metric_counts = {metric: 0 for metric in metrics}

            # Read jsonl file line by line
            with open(filepath, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    
                    # Iterate 4 metrics for accumulation
                    for metric in metrics:
                        if metric in data:
                            score_dict = data[metric]
                            s1 = score_dict.get('score1')
                            s2 = score_dict.get('score2')
                            
                            # Only add score and increment count when score is not None (filter null)
                            if s1 is not None:
                                metric_sums[metric] += float(s1)
                                metric_counts[metric] += 1
                                
                            if s2 is not None:
                                metric_sums[metric] += float(s2)
                                metric_counts[metric] += 1

            # Calculate average scores of four metrics for current file
            row_data = [filename]
            for metric in metrics:
                count = metric_counts[metric]
                if count > 0:
                    # Calculate average, keep 4 decimal places (adjustable)
                    avg = metric_sums[metric] / count
                    row_data.append(round(avg, 4))
                else:
                    # If a metric has all null values across 512 entries, prevent division-by-zero error
                    row_data.append('N/A')
            
            # Write one row of data for this file
            writer.writerow(row_data)

    print(f"Processing complete! Results saved to {output_csv}")

if __name__ == '__main__':
    analyze_scores()