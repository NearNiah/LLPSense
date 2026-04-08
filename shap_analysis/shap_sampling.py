"""
Sample top-N T5 feature indices by SHAP importance from human_shap.xlsx.

For each N in N_COMPONENTS_LIST, selects the top-N T5_* features ranked by
mean absolute SHAP value and saves their numeric indices to feature_index_N.txt.

Usage:
    python shap_sampling.py [--shap_file human_shap.xlsx] [--output_dir .]
"""

import argparse
import os

import pandas as pd

SHAP_FILE = os.path.join(os.path.dirname(__file__), "human_shap.xlsx")
N_COMPONENTS_LIST = [512, 256, 128, 64, 32, 16]
DEFAULT_OUTPUT_DIR = os.path.dirname(__file__)

# T5-0 to T5-1023 correspond to columns 0-1023 in the original feature matrix.

def load_t5_importance(shap_file: str) -> pd.Series:
    """Return T5_* features sorted by mean absolute SHAP value (descending).

    Reads the 'feature_importance' sheet which has columns [feature, mean_abs_shap].
    """
    df = pd.read_excel(shap_file, sheet_name="feature_importance")
    t5_mask = df["feature"].astype(str).str.startswith("T5_")
    t5_df = df[t5_mask].copy()
    if t5_df.empty:
        raise ValueError(f"No T5_* rows found in 'feature_importance' sheet of {shap_file}")
    importance = t5_df.set_index("feature")["mean_abs_shap"].sort_values(ascending=False)
    return importance


def main():
    parser = argparse.ArgumentParser(
        description="Save top-N T5 feature indices ranked by SHAP importance"
    )
    parser.add_argument("--shap_filename", default="human_shap.xlsx",
                        help="Name of shap file (human_shap.xlsx or llpsdb_shap.xlsx, default: %(default)s)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save feature_index_N.txt files (default: %(default)s)")
    args = parser.parse_args()
    shap_file = os.path.join(args.output_dir, args.shap_filename)

    tag = os.path.basename(args.shap_filename).split('_')[0]
    importance = load_t5_importance(shap_file)
    total = len(importance)
    print(f"Loaded {total} T5 features from {shap_file}")

    os.makedirs(args.output_dir, exist_ok=True)

    for n in N_COMPONENTS_LIST:
        if n > total:
            print(f"  [WARNING] N={n} > available T5 features ({total}), using all {total}")
            top_n = importance
        else:
            top_n = importance.head(n)

        indices = [int(col.split("_")[1]) for col in top_n.index]

        out_path = os.path.join(args.output_dir, f"{tag}_feature_index_{n}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(map(str, indices)) + "\n")

        print(f"  [N={n:>3d}] saved {len(indices)} indices → {out_path}")


if __name__ == "__main__":
    main()
