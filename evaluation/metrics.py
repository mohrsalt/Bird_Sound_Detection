import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import torch
from typing import List, Tuple, Union
import pandas as pd
from types import SimpleNamespace
# Each interval: (start_time, end_time)
# Each scored interval: (start_time, end_time, score in [0,1])
config = SimpleNamespace(
            fps=100, #here
            sample_rate=32000, #here
            onset=0.0, #here
            offset=0.0, #here
            padding=(0.0, 0.0), #here
            min_duration=0.0, #here
            min_silence=0.0, #here
        )
def intervals(
        predictions, audio_length
    ):
        
        intervals = predictions_to_intervals(predictions, config.fps)

        filtered_intervals = []
        for start, end in intervals:
            # Remove intervals that are too short
            if end - start < config.min_duration:
                continue
            else:
                filtered_intervals.append((start, end))

        # Pad intervals and check that intervals do not overlap with onset and offset
        audio_length = audio_length 
        padded_intervals = []
        if filtered_intervals:
            for start, end in filtered_intervals:
                # Add padding
                padded_start = max(0, start - config.padding[0])
                padded_end = min(audio_length, end + config.padding[1])

                # Trim at onset boundary
                if config.onset > 0:
                    if padded_end <= config.onset:
                        continue  # Skip if entirely in onset
                    padded_start = max(padded_start, config.onset)

                # Trim at offset boundary
                if config.offset > 0:
                    offset_start = audio_length - config.offset
                    if padded_start >= offset_start:
                        continue  # Skip if entirely in offset
                    padded_end = min(padded_end, offset_start)

                padded_intervals.append((padded_start, padded_end))

        # Merge intervals that are too close
        merged_intervals = []
        if padded_intervals:
            if config.min_silence > 0:
                current_interval = padded_intervals[0]
                for interval in padded_intervals[1:]:
                    if interval[0] - current_interval[1] < config.min_silence:
                        current_interval = (
                            current_interval[0],
                            max(current_interval[1], interval[1]),
                        )
                    else:
                        merged_intervals.append(current_interval)
                        current_interval = interval
                merged_intervals.append(current_interval)
            else:
                merged_intervals = padded_intervals

	
        
        return merged_intervals
def predictions_to_intervals(
        bool_array: torch.Tensor, fps: int
    ) -> List[Tuple[float, float]]:
        """
        Converts a boolean tensor array of predictions into a list of time intervals.
        This function identifies contiguous sequences of True values in the boolean array
        and converts them into time intervals based on the given frames per second (fps).

        Args:
            bool_array (torch.Tensor): A 1D boolean tensor where True values represent active segments
            fps (int): Frames per second, used to convert frame indices to time in seconds

        Returns:
            List[Tuple[float, float]]: A list of tuples where each tuple contains
                (start_time, end_time) in seconds for each detected interval
        """

        # Get indices where values change
        if bool_array.dim() == 0:
            bool_array = bool_array.unsqueeze(0)
        changes = torch.where(bool_array[:-1] != bool_array[1:])[0] + 1

        # Add start and end indices
        if bool_array[0]:
            changes = torch.cat([torch.tensor([0]).to(changes.device), changes])
        if bool_array[-1]:
            changes = torch.cat(
                [changes, torch.tensor([len(bool_array)]).to(changes.device)]
            )

        # Convert to intervals
        intervals = []
        for start, end in zip(changes[::2], changes[1::2]):
            # Convert frame indices to seconds (each frame is 10ms)
            start_time = float(start) / fps
            end_time = float(end) / fps
            intervals.append((start_time, end_time))
        return intervals

def merge_intervals(intervals):
    """Merge overlapping or adjacent intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

def interval_length(intervals):
    return sum(e - s for s, e in intervals)

def intersect_intervals(A, B):
    """Return list of intersections between interval lists A and B."""
    i, j = 0, 0
    intersections = []
    while i < len(A) and j < len(B):
        a1, a2 = A[i]
        b1, b2 = B[j]
        start, end = max(a1, b1), min(a2, b2)
        if start < end:
            intersections.append((start, end))
        if a2 < b2:
            i += 1
        else:
            j += 1
    return intersections

def union_length(A, B):
    """Length of union of two (already merged) interval lists."""
    return interval_length(merge_intervals(A + B))

# ---------------- TIME-BASED (continuous) ----------------

def compute_time_metrics(gt_segments, pred_segments, total_length):
    gt = merge_intervals(gt_segments)
    pred = merge_intervals(pred_segments)
    inter = merge_intervals(intersect_intervals(gt, pred))

    TP = interval_length(inter)
    FP = interval_length(pred) - TP
    FN = interval_length(gt) - TP
    TN = total_length - union_length(gt, pred)

    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (TP + TN) / total_length if total_length > 0 else 0.0

    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)

def compute_time_metrics_multi(gt_list, pred_list, total_lengths):
    """Micro-average across files by summing durations."""
    assert len(gt_list) == len(pred_list) == len(total_lengths)
    TP = FP = FN = TN = 0.0
    T = sum(total_lengths)
    for gt, pred, L in zip(gt_list, pred_list, total_lengths):
        m = compute_time_metrics(gt, pred, L)
        TP += m["TP"]; FP += m["FP"]; FN += m["FN"]; TN += m["TN"]
    precision = TP / (TP + FP) if TP + FP > 0 else 1.0
    recall = TP / (TP + FN) if TP + FN > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (TP + TN) / T if T > 0 else 0.0
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)

def time_based_pr_multi(gt_list, prepreds,logits_main, total_lengths):
    """
    Build time-based PR and ROC across files.
    At each global threshold, keep segments with score>=t in each file,
    then sum durations to get TP/FP/FN/TN and compute precision/recall/FPR/TPR.
    """
    assert len(gt_list) == len(prepreds) == len(logits_main) == len(total_lengths)

    thresholds = torch.arange(0.0, 1.0, 0.1)

    recalls, precisions, fprs, tprs = [], [], [], []
    
    for t in thresholds:
        count=0
        TP = FP = FN = TN = 0.0
        for gt, preds_scored,logits, L in zip(gt_list, prepreds,logits_main, total_lengths):
            
            print(count)
            count=count+1
           
            gt_m = merge_intervals(gt)
            
            logits=torch.tensor(logits)
            probs = torch.sigmoid(logits)
           
            predictions = probs > t


            pred_m=intervals(predictions, L)
            inter = intersect_intervals(gt_m, pred_m)

            tp = interval_length(inter)
            p_len = interval_length(pred_m)
            g_len = interval_length(gt_m)
            fp = p_len - tp
            fn = g_len - tp
            tn = L - union_length(gt_m, pred_m)

            TP += tp; FP += fp; FN += fn; TN += tn

        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        tpr = recall  # duration-based TPR equals recall

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)
        tprs.append(tpr)

    # Add endpoints for curve completeness (optional)
    if len(recalls) == 0 or recalls[-1] != 0.0:
        recalls.append(0.0); precisions.append(1.0)
    if (0.0, 0.0) not in zip(fprs, tprs):
        fprs.append(0.0); tprs.append(0.0)

    # Average Precision (area under P(R))
    order_pr = np.argsort(recalls)
    R_sorted = np.array(recalls)[order_pr]
    P_sorted = np.array(precisions)[order_pr]
    AP = float(np.trapz(P_sorted, R_sorted))  # trapezoid over recall

    # AUC-ROC
    order_roc = np.argsort(fprs)
    F_sorted = np.array(fprs)[order_roc]
    T_sorted = np.array(tprs)[order_roc]
    AUC = float(np.trapz(T_sorted, F_sorted))

    return dict(recalls=np.array(recalls),
                precisions=np.array(precisions),
                AP=AP,
                fprs=np.array(fprs),
                tprs=np.array(tprs),
                AUC=AUC)

if __name__ == "__main__":
    
    data=pd.read_csv("JavadPreds_top20.csv")

    data["Hypotheses"] = data["Hypotheses"].fillna("[]").apply(ast.literal_eval)
    data["Reference"]  = data["Reference"].fillna("[]").apply(ast.literal_eval)
    data["Logits"]  = data["Logits"].fillna("[]").apply(ast.literal_eval)
 
   

    gt_list=list(data["Reference"])
    prepreds=list(data["Hypotheses"])
    logs=list(data["Logits"])
    total_lengths=list(data["Length"])



    # Pred lists without scores
    pred_list = prepreds

    # ---- Time-based (micro across files)
    time_stats = compute_time_metrics_multi(gt_list, pred_list, total_lengths)
    curves_t = time_based_pr_multi(gt_list, prepreds,logs, total_lengths) #uncomment
    print("Time-based (micro):", time_stats)
    print(f"Time-based AP={curves_t['AP']:.3f}, AUC={curves_t['AUC']:.3f}") #uncomment


#uncomment whole thing below
    # ---- Plots
    # Time-based PR
    plt.figure(figsize=(6, 4))
    plt.plot(curves_t["recalls"], curves_t["precisions"], marker='o')
    plt.xlabel("Recall (time)")
    plt.ylabel("Precision (time)")
    plt.title(f"Time-based PR (AP={curves_t['AP']:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_pr_curve.png", dpi=300)
    plt.show()

    # Time-based ROC
    plt.figure(figsize=(6, 4))
    plt.plot(curves_t["fprs"], curves_t["tprs"], marker='o')
    plt.xlabel("False Positive Rate (time)")
    plt.ylabel("True Positive Rate (time)")
    plt.title(f"Time-based ROC (AUC={curves_t['AUC']:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_roc_curve.png", dpi=300)
    plt.show()




