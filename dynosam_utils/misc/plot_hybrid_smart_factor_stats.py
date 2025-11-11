import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Optional: nicer style
sns.set(style="whitegrid")

def plot_reprojection_errors(csv_file):
    # Load the CSV
    df = pd.read_csv(csv_file)

    # Ensure correct types
    df['tracklet_id'] = df['tracklet_id'].astype(str)
    df['frame_id'] = df['frame_id'].astype(int)
    df['reprojection_error'] = df['reprojection_error'].astype(float)
    df['is_good'] = df['is_good'].astype(bool)

    # Plot setup
    plt.figure(figsize=(12, 6))

    # Plot one line per tracklet
    for tracklet_id, group in df.groupby('tracklet_id'):
        label = f"Tracklet {tracklet_id}"
        plt.plot(group['frame_id'], group['reprojection_error'], label=label, alpha=0.7)

    plt.xlabel("Frame ID")
    plt.ylabel("Reprojection Error")
    plt.title("Reprojection Error Over Time per Tracklet")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), ncol=1, fontsize='small')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot reprojection errors per tracklet.")
    parser.add_argument("csv_file", help="Path to input CSV file.")
    args = parser.parse_args()

    plot_reprojection_errors(args.csv_file)
