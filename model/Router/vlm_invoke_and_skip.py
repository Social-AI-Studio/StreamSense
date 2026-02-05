import argparse
import pandas as pd
import os
import os.path as osp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ----- Argument Parser -----
def parse_args():
    parser = argparse.ArgumentParser(description="Encoder + VLM Fusion Evaluation")
    
    parser.add_argument("--dataset_name", type=str, default="MOSI", choices=["MOSI", "MOSEI", "HateBit"], help="Dataset name")
    parser.add_argument("--dataset_type", type=str, default="test", help="Dataset split (train/val/test)")
    parser.add_argument("--modality", type=str, default="vta", help="Input modality")
    parser.add_argument("--data_dir", type=str, default="/mnt/sda/wanghan/ICLR2026/fuse_model/dataset", help="Directory of CSV files")
    parser.add_argument("--MaxDefer", type=int, default=6, help="Maximal defer step allowed")
    parser.add_argument("--MaxEnc_ratio", type=float, default=3.0, help="Maximal encoder prediction step allowed")
    
    parser.add_argument("--encoder_csv", type=str, required=True, help="Encoder prediction CSV")
    parser.add_argument("--vlm_csv", type=str, required=True, help="VLM prediction CSV")
    parser.add_argument("--root", type=str, default="./model/Router/Prediction", help="Root folder")


    return parser.parse_args()

# ----- Metric Evaluation Function -----
def evaluate_model(preds, gt, dataset_name):
    acc = accuracy_score(gt, preds)
    f1_macro = f1_score(gt, preds, average='macro')
    
    pos_label = "Offensive" if dataset_name.lower() == "hatebit" else "Positive"
    
    f1_pos = f1_score(gt, preds, pos_label=pos_label)
    precision_pos = precision_score(gt, preds, pos_label=pos_label)
    recall_pos = recall_score(gt, preds, pos_label=pos_label)
    
    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "f1_Positive": f1_pos,
        "precision_Positive": precision_pos,
        "recall_Positive": recall_pos
    }


# ----- Match Encoder & VLM Predictions -----
def match_predictions(df_vlm, df_enc, positive_value, negative_value):
    matched_results = []
    for _, row in df_vlm.iterrows():
        video_index = row['video_id']
        time_str = row['cur_time_id']
        label = row["gt"]
        time = float(time_str)
        match_row = df_enc[(df_enc['video_id'] == float(video_index)) & 
                           (df_enc['cur_time_id'] == time)]
        if len(match_row) == 0:
            continue

        # VLM prediction
        v_pred = row["pred"]
        if(positive_value in v_pred):
            v_pred = positive_value
            v_prob = row["prob_pos"]
        else:
            v_pred = negative_value
            v_prob = row["prob_neg"]

        # Encoder prediction
        enc_row = match_row.iloc[0]
        if enc_row['prob_pos'] > enc_row['prob_neg']:
            t_pred = positive_value
            t_prob = enc_row['prob_pos']
        else:
            t_pred = negative_value
            t_prob = enc_row['prob_neg']

        matched_results.append({
            'video_id': video_index,
            'cur_time_id': time_str,
            "encoder_prediction": t_pred,
            "encoder_probability": t_prob,
            "vlm_prediction": v_pred,
            "vlm_probability": v_prob,
            "gt_label": label
        })
        df = pd.DataFrame(matched_results)
      
    return df

# ----- Fusion Algorithm -----
def fusion_prediction(df_matched, MaxDefer, MaxEnc_ratio):
    MaxEnc = int((MaxDefer) * MaxEnc_ratio)
    fusion_preds, fusion_gts, actions = [], [], []
    current_video, t_suc, start_time, current_label = None, None, -1, None

    for idx, row in df_matched.iterrows():
        vid, t = row["video_id"],  row["cur_time_id"]

        if current_video != vid:
            current_video, t_suc, t_suc, start_time = vid, t, t, t

        d = t - t_suc
        small_thresh = 1.0 if t == start_time else 0.5 + 0.5 * min(d, MaxEnc) / (MaxEnc)
        large_thresh = 0.5 if t == start_time else 1.0 - 0.5 * min(d, MaxDefer) / (MaxDefer)
            
        # Check small model
        if row["encoder_probability"] >= small_thresh and row["encoder_prediction"] == current_label:
            fusion_preds.append(row["encoder_prediction"])
            fusion_gts.append(row["gt_label"])
            actions.append("encoder")
            
            t_suc = t
        elif row["vlm_probability"] >= large_thresh:
            fusion_preds.append(row["vlm_prediction"])
            fusion_gts.append(row["gt_label"])
            actions.append("vlm_invoke")

            t_suc = t
            current_label = row["vlm_prediction"]
        else:
            fusion_preds.append("None")
            fusion_gts.append(row["gt_label"])
            actions.append("defer")
               
    df_matched['fusion_prediction'] = fusion_preds
    df_matched['action'] = actions
    return df_matched

# ----- Main Function -----
def main():
    args = parse_args()

    positive_value = "Positive" if args.dataset_name not in ["HateBit"] else "Offensive"
    negative_value = "Negative" if args.dataset_name not in ["HateBit"] else "Normal"

    # Load prediction CSVs
    df_enc = pd.read_csv(args.encoder_csv)
    df_vlm = pd.read_csv(args.vlm_csv)

    df_matched = match_predictions(df_vlm, df_enc, positive_value, negative_value)
    df_fused = fusion_prediction(df_matched, args.MaxDefer, args.MaxEnc_ratio)

    # Save final output
    output_csv = osp.join(args.data_dir, f'{args.dataset_name.lower()}_{args.modality}_fusion_output.csv')
    df_fused.to_csv(output_csv, index=False)
    print(f"Saved fused predictions to {output_csv}")

    # Evaluate
    results = evaluate_model(df_fused['encoder_prediction'], df_fused['gt_label'], args.dataset_name)
    print("\nEncoder Model Evaluation:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    
    results = evaluate_model(df_fused['vlm_prediction'], df_fused['gt_label'], args.dataset_name)
    print("\nVLM Model Evaluation:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    df_eval = df_fused[df_fused['fusion_prediction'] != "None"]
    print(len(df_eval))
    results = evaluate_model(df_eval['fusion_prediction'], df_eval['gt_label'], args.dataset_name)
    print("\nFusion Model Evaluation:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

     # ----- Print Action Percentages -----
    total_samples = len(df_fused)
    vlm_count = sum(df_fused['action']== "vlm_invoke")
    defer_count = sum(df_fused['action'] == "defer")
    print(f"\nAction Percentages:")
    print(f"  Number of Timestamp: {total_samples}")
    print(f"  VLM invoked: {vlm_count/total_samples:.2%}")
    print(f"  defer: {defer_count/total_samples:.2%}")

    
    os.makedirs(f'{args.root}/{args.dataset_name}', exist_ok=True)
    save_path = os.path.join(f'{args.root}/{args.dataset_name}', f"modality_{args.modality}_maxdefer_{args.MaxDefer}_maxenc_{int(args.MaxEnc_ratio*args.MaxDefer)}_{args.dataset_type}.csv")
    df_fused.to_csv(save_path, index=False)
    print(f"Fusion predictions saved to: {save_path}")

if __name__ == "__main__":
    main()
