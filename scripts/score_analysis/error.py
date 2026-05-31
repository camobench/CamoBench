import os
import json
import csv

def get_error_type(IF, OVQ, SCSC, SCISC):
    """
    Determine error type by priority 1 to 8. Return the matching type immediately.
    Return None if no type matches.
    """
    # Convert to integers for comparison to avoid comparing string "1" with int 1
    try:
        IF = int(IF)
        OVQ = int(OVQ)
        SCSC = int(SCSC)
        SCISC = int(SCISC)
    except (ValueError, TypeError):
        return None # Skip if there are missing values that cannot be converted

    # Type 1: Severe Structural Corruption
    if OVQ == 1:
        return 1
    # Type 2: Scene Semantic Mismatch
    elif SCSC == 0 and OVQ != 1:
        return 2
    # Type 3: Concept Omission
    elif IF == 1:
        return 3
    # Type 4: Illusion Structural Deviation
    elif SCISC == 2:
        return 4
    # Type 5: Blatant Pasting Effect
    elif IF == 2 and OVQ in [2, 3]:
        return 5
    # Type 6: Independent Object Manifestation
    elif IF == 2 and OVQ in [4, 5]:
        return 6
    # Type 7: Traceable Manipulation
    elif OVQ == 3:
        return 7
    # Type 8: Suboptimal Camouflage
    elif IF == 3:
        return 8
    
    return None

def process_scores():
    input_dir = os.path.join("data", "score")
    output_dir = os.path.join("data", "score_analysis")
    output_csv = os.path.join(output_dir, "error_type_statistics.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare result list for CSV writing
    results = []

    # Iterate all jsonl files
    if not os.path.exists(input_dir):
        print(f"Cannot find input directory: {input_dir}")
        return

    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"):
            continue
            
        filepath = os.path.join(input_dir, filename)
        
        # Initialize 8 type counters for this file
        counts = {f"Type {i}": 0 for i in range(1, 9)}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                data = json.loads(line.strip())
                
                # Get the dictionary for each metric
                # Note: Ensure the dictionary keys here match exactly with your jsonl (including full-width/half-width dashes)
                IF_dict = data.get("Illusion Fidelity", {})
                OVQ_dict = data.get("Overall Visual Quality", {})
                SCSC_dict = data.get("Semantic Consistency – Scene Consistency", {})
                SCISC_dict = data.get("Semantic Consistency – Illusion Shape Consistency", {})
                
                # -------------------------
                # Data 1: all using score1
                # -------------------------
                type_data1 = get_error_type(
                    IF=IF_dict.get("score1"),
                    OVQ=OVQ_dict.get("score1"),
                    SCSC=SCSC_dict.get("score1"),
                    SCISC=SCISC_dict.get("score1")
                )
                if type_data1 is not None:
                    counts[f"Type {type_data1}"] += 1

                # -------------------------
                # Data 2: use score1 for Overall Visual Quality, use score2 for the rest
                # -------------------------
                type_data2 = get_error_type(
                    IF=IF_dict.get("score2"),
                    OVQ=OVQ_dict.get("score1"),  # Per your rule, since score2 is null, use score1 here
                    SCSC=SCSC_dict.get("score2"),
                    SCISC=SCISC_dict.get("score2")
                )
                if type_data2 is not None:
                    counts[f"Type {type_data2}"] += 1
        
        # Add current file's statistics to result list
        row_result = {"Filename": filename}
        row_result.update(counts)
        results.append(row_result)

    # Write results to CSV file
    if results:
        fieldnames = ["Filename"] + [f"Type {i}" for i in range(1, 9)]
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Statistics complete! Processed {len(results)} files, results saved to: {output_csv}")
    else:
        print("No jsonl files found or processing result is empty.")

if __name__ == "__main__":
    process_scores()