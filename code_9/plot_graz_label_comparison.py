import os
import numpy as np
import matplotlib.pyplot as plt

def load_labels(label_type: str, label_metric: str) -> np.ndarray:
    """
    Loads and concatenates .npy files for a specific label metric from the ImagEEG dataset.
    
    Args:
        label_type (str): The type of label, 'PUBLIC' or 'PRIVATE'.
        label_metric (str): The name of the metric (e.g., 'arousal_public', 'rate_valence_private').

    Returns:
        np.ndarray: A numpy array containing all concatenated labels.
                    Returns an empty array if no files are found.
    """
    # The path points to the segmented data saved on disk
    data_dir = os.path.join(".", "labels","graz", label_type.upper())
    
    if not os.path.isdir(data_dir):
        print(f"ERROR: Folder '{os.path.abspath(data_dir)}' not found.")
        return np.array([])

    all_labels = []
    files_found = 0
    
    print(f"Searching files for '{label_metric}' in: {os.path.abspath(data_dir)}")

    # The file name is in the format {subject_id}_{label_metric}.npy
    file_suffix = f"_{label_metric}.npy"

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(file_suffix):
            file_path = os.path.join(data_dir, filename)
            try:
                data = np.load(file_path)
                all_labels.append(data)
                files_found += 1
                print(f"  -> Loaded: {filename} (Shape: {data.shape})")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")

    if not all_labels:
        print(f"No files found for metric '{label_metric}' in '{data_dir}'.")
        return np.array([])
    
    concatenated_labels = np.concatenate(all_labels)
    print(f"Found and concatenated {files_found} files for '{label_metric}'. Final Shape: {concatenated_labels.shape}")
    return concatenated_labels

def plot_comparison(metric_name: str, public_labels: np.ndarray, private_labels: np.ndarray, plots_dir: str):
    """
    Creates and saves a side-by-side comparative plot for public and private label distributions.
    Input labels must already be rounded.

    Args:
        metric_name (str): The name of the metric to display (e.g., 'Arousal', 'Valence').
        public_labels (np.ndarray): Array of rounded public labels.
        private_labels (np.ndarray): Array of rounded private labels.
        plots_dir (str): The folder where the generated plot will be saved.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # --- Plot for public labels ---
    min_val_pub = int(public_labels.min())
    max_val_pub = int(public_labels.max())
    bins_pub = np.arange(min_val_pub - 0.5, max_val_pub + 1.5, 1)
    ax1.hist(public_labels, bins=bins_pub, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title(f'Public Distribution ({metric_name})', fontsize=16)
    ax1.set_xlabel(f'Rounded {metric_name} Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_xticks(range(min_val_pub, max_val_pub + 1))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot for private labels ---
    min_val_priv = int(private_labels.min())
    max_val_priv = int(private_labels.max())
    bins_priv = np.arange(min_val_priv - 0.5, max_val_priv + 1.5, 1)
    ax2.hist(private_labels, bins=bins_priv, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.set_title(f'Private Distribution ({metric_name})', fontsize=16)
    ax2.set_xlabel(f'Rounded {metric_name} Value', fontsize=12)
    ax2.set_xticks(range(min_val_priv, max_val_priv + 1))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle(f'Comparison of Rounded {metric_name} Distributions: Public vs. Private (ImagEEG)', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_filename = os.path.join(plots_dir, f'ImagEEG_comparison_rounded_{metric_name.lower()}.png')
    plt.savefig(plot_filename)
    print(f"\nComparative plot for {metric_name} saved to: {plot_filename}")
    plt.show()

if __name__ == '__main__':
    # Folder to save plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # --- 1. Arousal Comparison ---
    print("\n##### Starting processing for AROUSAL #####")
    arousal_pub_raw = load_labels('PUBLIC', 'arousal_pubblico') # Keep original metric names as they refer to filenames
    arousal_priv_raw = load_labels('PRIVATE', 'rate_arousal_privato') # Keep original metric names as they refer to filenames

    if arousal_pub_raw.size > 0 and arousal_priv_raw.size > 0:
        plot_comparison('Arousal', np.round(arousal_pub_raw), np.round(arousal_priv_raw), plots_dir)

    # --- 2. Valence Comparison ---
    print("\n\n##### Starting processing for VALENCE #####")
    valence_pub_raw = load_labels('PUBLIC', 'valence_pubblico') # Keep original metric names as they refer to filenames
    valence_priv_raw = load_labels('PRIVATE', 'rate_valence_privata') # Keep original metric names as they refer to filenames

    if valence_pub_raw.size > 0 and valence_priv_raw.size > 0:
        plot_comparison('Valence', np.round(valence_pub_raw), np.round(valence_priv_raw), plots_dir)

    print("\n\nProcessing complete.")