import os, glob, json
import pandas as pd
import matplotlib.pyplot as plt

def analyze_param(base_dir, param_type, output_dir="results/SuperParam"):
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(base_dir, f"ourmodel-{param_type}_*")
    param_dirs = glob.glob(pattern)

    records = []
    for d in param_dirs:
        param_value = d.split(f"ourmodel-{param_type}_")[-1]

        for fold in os.listdir(d):
            metrics_file = os.path.join(d, fold, "test_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                metrics["ParamValue"] = param_value
                metrics["Fold"] = fold
                records.append(metrics)
    
    if not records:
        print(f"[WARN] No results found for param_type={param_type}")
        return None
    
    df = pd.DataFrame(records)

    try:
        df["ParamValue"] = df["ParamValue"].astype(float)
    except:
        pass  

    mean_df = df.groupby("ParamValue").mean(numeric_only=True)
    std_df  = df.groupby("ParamValue").std(numeric_only=True)

    summary = pd.concat(
        [mean_df.add_suffix("_mean"), std_df.add_suffix("_std")], axis=1
    ).reset_index()

    plt.figure(figsize=(10,6))
    metrics = ["AUC", "AUPR", "F1", "Precision", "Recall", "Accuracy"]

    for metric in metrics:
        plt.errorbar(
            summary["ParamValue"],
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"],
            marker="o", capsize=5, label=metric
        )

    plt.xlabel(param_type)
    plt.ylabel("Score")
    plt.title(f"Model Performance vs {param_type} (mean ± std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"{param_type}_plot.png")
    csv_path = os.path.join(output_dir, f"{param_type}_summary.csv")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    summary.to_csv(csv_path, index=False)

    best_row = summary.loc[summary["AUC_mean"].idxmax()]
    best_value = best_row["ParamValue"]
    best_auc   = best_row["AUC_mean"]

    print(f"[OK] {param_type}: 推荐值={best_value}, AUC={best_auc:.4f}")
    return {"ParamType": param_type, "BestValue": best_value, "BestAUC": best_auc}



base_dir = "results"
# param_types = ["ContrastiveTemp", "Dropout", "Epoch", "MetaLambda", "WeightDecay", "LR"]
param_types = ["MetaLambda"]
recommendations = []
for p in param_types:
    rec = analyze_param(base_dir, p)
    if rec:
        recommendations.append(rec)

rec_df = pd.DataFrame(recommendations)
rec_df.to_csv("results/SuperParam/BestParams.csv", index=False)
print(rec_df)
