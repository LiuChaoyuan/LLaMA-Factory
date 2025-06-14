# utils.py
import json
import difflib
from collections import defaultdict
from tqdm import tqdm

def parse_quadruplets(output_str: str):
    """
    Parses the model output string into a list of quadruplets.
    Example: "Target1 | Arg1 | Group1 | hate [SEP] Target2 | Arg2 | Group2 | non-hate [END]"
    """
    quadruplets = []
    if not output_str or output_str.strip() == "":
        return quadruplets

    # Remove [END] and split by [SEP]
    if output_str.endswith("[END]"):
        output_str = output_str[:-5].strip() # Remove "[END]" and potential trailing space
    
    parts = output_str.split("[SEP]")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        elements = [e.strip() for e in part.split("|")]
        if len(elements) == 4:
            quadruplets.append(tuple(elements))
        else:
            # Handle potential malformed output, maybe log it
            # print(f"Warning: Malformed quadruplet part: '{part}' from '{output_str}'")
            # For robustness, we might try to pad or skip
            # For now, we'll strictly expect 4 elements
            pass 
            
    return quadruplets

def calculate_similarity(s1: str, s2: str) -> float:
    """
    Calculates similarity between two strings using SequenceMatcher.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_f1_scores(pred_quads_list, gold_quads_list, similarity_threshold=0.5):
    """
    Computes hard and soft F1 scores for a batch of predictions.
    
    Args:
        pred_quads_list: A list of lists, where each inner list contains predicted quadruplets (tuples) for one sample.
        gold_quads_list: A list of lists, where each inner list contains gold quadruplets (tuples) for one sample.
        similarity_threshold: The threshold for soft matching Target and Argument.

    Returns:
        A dictionary with 'f1_hard', 'p_hard', 'r_hard', 'f1_soft', 'p_soft', 'r_soft'.
    """
    tp_hard, fp_hard, fn_hard = 0, 0, 0
    tp_soft, fp_soft, fn_soft = 0, 0, 0

    for pred_quads, gold_quads in zip(pred_quads_list, gold_quads_list):
        # --- Hard Match ---
        current_tp_hard = 0
        # Make copies to allow removal
        temp_gold_quads_hard = list(gold_quads) 
        
        for pq in pred_quads:
            if pq in temp_gold_quads_hard:
                current_tp_hard += 1
                temp_gold_quads_hard.remove(pq) # Remove matched gold to avoid double counting
            else:
                fp_hard +=1 # Predicted, but not in gold

        tp_hard += current_tp_hard
        fn_hard += len(temp_gold_quads_hard) # Gold quads not matched by any prediction

        # --- Soft Match ---
        current_tp_soft = 0
        # Make copies to allow removal
        temp_gold_quads_soft = list(gold_quads)
        
        for pq_target, pq_arg, pq_group, pq_hateful in pred_quads:
            matched_soft = False
            best_match_gold_idx = -1
            
            for i, (gq_target, gq_arg, gq_group, gq_hateful) in enumerate(temp_gold_quads_soft):
                if pq_group == gq_group and pq_hateful == gq_hateful:
                    sim_target = calculate_similarity(pq_target, gq_target)
                    sim_arg = calculate_similarity(pq_arg, gq_arg)
                    if sim_target >= similarity_threshold and sim_arg >= similarity_threshold:
                        # Found a soft match - for simplicity, take the first one
                        # A more sophisticated approach might find the *best* soft match
                        # if multiple exist, but this is usually sufficient.
                        best_match_gold_idx = i
                        break 
            
            if best_match_gold_idx != -1:
                current_tp_soft +=1
                del temp_gold_quads_soft[best_match_gold_idx] # Remove matched gold
                matched_soft = True

            if not matched_soft:
                fp_soft += 1 # Predicted, but no soft match in gold

        tp_soft += current_tp_soft
        fn_soft += len(temp_gold_quads_soft) # Gold quads not soft-matched

    # Calculate P, R, F1 for hard match
    p_hard = tp_hard / (tp_hard + fp_hard) if (tp_hard + fp_hard) > 0 else 0
    r_hard = tp_hard / (tp_hard + fn_hard) if (tp_hard + fn_hard) > 0 else 0
    f1_hard = 2 * p_hard * r_hard / (p_hard + r_hard) if (p_hard + r_hard) > 0 else 0

    # Calculate P, R, F1 for soft match
    p_soft = tp_soft / (tp_soft + fp_soft) if (tp_soft + fp_soft) > 0 else 0
    r_soft = tp_soft / (tp_soft + fn_soft) if (tp_soft + fn_soft) > 0 else 0
    f1_soft = 2 * p_soft * r_soft / (p_soft + r_soft) if (p_soft + r_soft) > 0 else 0
    
    f1_avg = (f1_hard + f1_soft) / 2

    return {
        "p_hard": p_hard, "r_hard": r_hard, "f1_hard": f1_hard,
        "p_soft": p_soft, "r_soft": r_soft, "f1_soft": f1_soft,
        "f1_avg": f1_avg
    }

def load_json_data(file_path, max_samples=None, is_train=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if max_samples:
        raw_data = raw_data[:max_samples]
        
    for item in tqdm(raw_data, desc=f"Loading {file_path}"):
        entry = {'id': item['id'], 'content': item['content']}
        if is_train:
            entry['output'] = item['output']
        data.append(entry)
    return data