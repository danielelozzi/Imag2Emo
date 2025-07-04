import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as patches # Added for plotting circles

# Funzioni load_labels dai file originali per DEAP e Graz
def load_labels_deap(label_type: str, label_metric: str) -> np.ndarray:
    """
    Loads and concatenates .npy files for a specific label metric from the DEAP dataset.
    """
    data_dir = os.path.join(".", "labels", "deap", label_type.upper())
    
    if not os.path.isdir(data_dir):
        print(f"ERROR: Folder '{os.path.abspath(data_dir)}' not found.")
        return np.array([])

    all_labels = []
    
    file_suffix = f"_{label_metric}.npy"

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(file_suffix):
            file_path = os.path.join(data_dir, filename)
            try:
                data = np.load(file_path)
                all_labels.append(data)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")

    if not all_labels:
        print(f"No files found for metric '{label_metric}' in '{data_dir}'.")
        return np.array([])
    
    concatenated_labels = np.concatenate(all_labels)
    return concatenated_labels

def load_labels_graz(label_type: str, label_metric: str) -> np.ndarray:
    """
    Loads and concatenates .npy files for a specific label metric from the ImagEEG dataset.
    """
    data_dir = os.path.join(".", "labels","graz", label_type.upper())
    
    if not os.path.isdir(data_dir):
        print(f"ERROR: Folder '{os.path.abspath(data_dir)}' not found.")
        return np.array([])

    all_labels = []
    
    file_suffix = f"_{label_metric}.npy"

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(file_suffix):
            file_path = os.path.join(data_dir, filename)
            try:
                data = np.load(file_path)
                all_labels.append(data)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")

    if not all_labels:
        print(f"No files found for metric '{label_metric}' in '{data_dir}'.")
        return np.array([])
    
    concatenated_labels = np.concatenate(all_labels)
    return concatenated_labels

def plot_valence_arousal_comparison(
    dataset_name: str,
    public_arousal: np.ndarray, public_valence: np.ndarray,
    private_arousal: np.ndarray, private_valence: np.ndarray,
    plots_dir: str,
    num_samples: int = 10
):
    """
    Creates and saves a Valence-Arousal plane plot comparing public and private labels.
    Samples num_samples random indices from the available data, assigning unique colors
    to each pair and attempting to reduce coordinate overlap for sampled points.
    Includes thick black lines at the value of 5 on both axes and sets plot limits from 0 to 10.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    min_len = min(len(public_arousal), len(public_valence), len(private_arousal), len(private_valence))

    if min_len == 0:
        print(f"Skipping plot for {dataset_name}: Insufficient data.")
        plt.close(fig)
        return

    public_arousal = np.round(public_arousal[:min_len])
    public_valence = np.round(public_valence[:min_len])
    private_arousal = np.round(private_arousal[:min_len])
    private_valence = np.round(private_valence[:min_len])

    colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.XKCD_COLORS.keys())
    if num_samples > len(colors):
        colors = plt.cm.get_cmap('hsv', num_samples)
        colors = [colors(i) for i in range(num_samples)]
    else:
        colors = colors[:num_samples]

    all_indices = np.arange(min_len)
    np.random.shuffle(all_indices)

    selected_indices = []
    sampled_coords = set()

    for idx in all_indices:
        pub_coord = (public_valence[idx], public_arousal[idx])
        priv_coord = (private_valence[idx], private_arousal[idx])

        if pub_coord not in sampled_coords and priv_coord not in sampled_coords:
            selected_indices.append(idx)
            sampled_coords.add(pub_coord)
            sampled_coords.add(priv_coord)
            if len(selected_indices) == num_samples:
                break
    
    if len(selected_indices) < num_samples:
        print(f"Warning: Could not find {num_samples} non-overlapping coordinate pairs. Selected {len(selected_indices)}.")
        remaining_samples = num_samples - len(selected_indices)
        if remaining_samples > 0:
            additional_indices = np.random.choice(list(set(all_indices) - set(selected_indices)), remaining_samples, replace=False)
            selected_indices.extend(additional_indices)

    jitter_scale = 0.1
    
    for i, idx in enumerate(selected_indices):
        color = colors[i % len(colors)]

        pub_val_jittered = public_valence[idx] + np.random.uniform(-jitter_scale, jitter_scale)
        pub_arou_jittered = public_arousal[idx] + np.random.uniform(-jitter_scale, jitter_scale)
        priv_val_jittered = private_valence[idx] + np.random.uniform(-jitter_scale, jitter_scale)
        priv_arou_jittered = private_arousal[idx] + np.random.uniform(-jitter_scale, jitter_scale)

        ax.scatter(pub_val_jittered, pub_arou_jittered,
                   marker='o', s=100, color=color, label=f'Public Sample {i+1}', alpha=0.8)
        ax.scatter(priv_val_jittered, priv_arou_jittered,
                   marker='s', s=100, color=color, label=f'Private Sample {i+1}' if i==0 else "", alpha=0.8)

        ax.plot([pub_val_jittered, priv_val_jittered],
                [pub_arou_jittered, priv_arou_jittered],
                color=color, linestyle='-', linewidth=1.5, alpha=0.6)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Public Points',
               markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Private Points',
               markerfacecolor='gray', markersize=10),
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Sample Connection', alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.set_title(f'Valence-Arousal Comparison: Public vs. Private ({dataset_name})', fontsize=16)
    ax.set_xlabel('Rounded Valence Value', fontsize=12)
    ax.set_ylabel('Rounded Arousal Value', fontsize=12)

    # Set x and y limits and ticks to 0-10 for all plots
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 11))

    # Add thick black lines at 5 on both axes
    ax.axvline(5, color='black', linestyle='-', linewidth=2)
    ax.axhline(5, color='black', linestyle='-', linewidth=2)

    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_filename = os.path.join(plots_dir, f'{dataset_name.lower().replace(" ", "_")}_VA_comparison_multicolor.png')
    plt.savefig(plot_filename)
    print(f"\nValence-Arousal plot for {dataset_name} saved to: {plot_filename}")
    plt.show()


def plot_sampled_mean_points(
    dataset_name: str,
    public_arousal: np.ndarray, public_valence: np.ndarray,
    private_arousal: np.ndarray, private_valence: np.ndarray,
    plots_dir: str,
    num_groups: int = 4, # Changed to 4 groups
    samples_per_group: int = 38 # Changed to 38 samples per group
):
    """
    Creates and saves a Valence-Arousal plane plot showing mean points for N sampled groups,
    for both public and private labels. Includes thick black lines at the value of 5 on both axes
    and sets plot limits from 0 to 10. A circle around each mean point indicates the standard deviation.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    min_len = min(len(public_arousal), len(public_valence), len(private_arousal), len(private_valence))

    if min_len < num_groups * samples_per_group:
        print(f"Skipping mean points plot for {dataset_name}: Not enough data for {num_groups} groups of {samples_per_group} samples each ({num_groups * samples_per_group} required, {min_len} available).")
        plt.close(fig)
        return

    # Generate distinct colors for each group
    group_colors = plt.cm.get_cmap('Dark2', num_groups) 
    group_colors = [group_colors(i) for i in range(num_groups)]

    # Keep track of used indices to ensure distinct groups
    all_available_indices = np.arange(min_len)
    np.random.shuffle(all_available_indices)
    used_indices = set()
    
    sampled_group_indices = []
    
    # Try to select non-overlapping indices for groups
    attempts = 0
    while len(sampled_group_indices) < num_groups and attempts < min_len * 2: # Limit attempts to prevent infinite loops
        candidate_indices = np.random.choice(all_available_indices, samples_per_group, replace=False)
        
        # Check if any of these indices have been used
        if not any(idx in used_indices for idx in candidate_indices):
            sampled_group_indices.append(candidate_indices)
            used_indices.update(candidate_indices)
        attempts += 1
    
    if len(sampled_group_indices) < num_groups:
        print(f"Warning: Could not find {num_groups} distinct groups of {samples_per_group} samples. Found {len(sampled_group_indices)}.")

    for i, indices in enumerate(sampled_group_indices):
        if i >= num_groups:
            break

        # Public Labels
        pub_vals = public_valence[indices]
        pub_arous = public_arousal[indices]
        mean_v_pub = np.mean(pub_vals)
        mean_a_pub = np.mean(pub_arous)
        # Calculate Euclidean distance-based standard deviation for the circle radius
        std_pub = np.sqrt(np.std(pub_vals)**2 + np.std(pub_arous)**2)
        
        # Private Labels (using the same indices)
        priv_vals = private_valence[indices]
        priv_arous = private_arousal[indices]
        mean_v_priv = np.mean(priv_vals)
        mean_a_priv = np.mean(priv_arous)
        std_priv = np.sqrt(np.std(priv_vals)**2 + np.std(priv_arous)**2)
        
        color = group_colors[i]

        # Plot public mean
        ax.plot(mean_v_pub, mean_a_pub, marker='o', markersize=10, color=color,
                label=f'Public Mean Group {i+1}', alpha=0.8)
        ax.text(mean_v_pub + 0.1, mean_a_pub + 0.1, f'P{i+1}', fontsize=9, color='black', ha='left', va='bottom')
        
        # Add circle for public std dev
        if std_pub > 0: # Only draw if there's variation
            public_circle = patches.Circle((mean_v_pub, mean_a_pub), radius=std_pub,
                                          edgecolor=color, facecolor=color, alpha=0.2, linewidth=1)
            ax.add_patch(public_circle)

        # Plot private mean
        ax.plot(mean_v_priv, mean_a_priv, marker='s', markersize=10, color=color,
                label=f'Private Mean Group {i+1}' if i==0 else "", alpha=0.8) # Label only for the first private group
        ax.text(mean_v_priv + 0.1, mean_a_priv + 0.1, f'Pr{i+1}', fontsize=9, color='black', ha='left', va='bottom')

        # Add circle for private std dev
        if std_priv > 0: # Only draw if there's variation
            private_circle = patches.Circle((mean_v_priv, mean_a_priv), radius=std_priv,
                                           edgecolor=color, facecolor=color, alpha=0.2, linewidth=1)
            ax.add_patch(private_circle)

        # Draw a line connecting public and private means for the same group
        ax.plot([mean_v_pub, mean_v_priv], [mean_a_pub, mean_a_priv],
                color=color, linestyle='--', linewidth=1.5, alpha=0.6)

    ax.set_title(f'Mean Valence-Arousal Points of Sampled Groups ({dataset_name})', fontsize=16)
    ax.set_xlabel('Mean Valence Value', fontsize=12)
    ax.set_ylabel('Mean Arousal Value', fontsize=12)

    # Set x and y limits and ticks to 0-10 for all plots
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 11))

    # Add thick black lines at 5 on both axes
    ax.axvline(5, color='black', linestyle='-', linewidth=2)
    ax.axhline(5, color='black', linestyle='-', linewidth=2)

    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Custom legend for markers and circles
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Public Mean', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Private Mean', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Group Connection', alpha=0.6),
        patches.Circle((0, 0), radius=0.5, edgecolor='gray', facecolor='gray', alpha=0.2,
                       label='Standard Deviation Circle') # Generic circle for legend
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)


    plt.tight_layout()
    plot_filename = os.path.join(plots_dir, f'{dataset_name.lower().replace(" ", "_")}_sampled_mean_points.png')
    plt.savefig(plot_filename)
    print(f"\nSampled mean points plot for {dataset_name} saved to: {plot_filename}")
    plt.show()

if __name__ == '__main__':
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # --- Processing for DEAP ---
    print("\n##### Processing for DEAP #####")
    arousal_pub_deap = load_labels_deap('PUBLIC', 'arousal_pubblica')
    valence_pub_deap = load_labels_deap('PUBLIC', 'valence_pubblica')
    arousal_priv_deap = load_labels_deap('PRIVATE', 'arousal_privata')
    valence_priv_deap = load_labels_deap('PRIVATE', 'valence_privata')

    if all(arr.size > 0 for arr in [arousal_pub_deap, valence_pub_deap, arousal_priv_deap, valence_priv_deap]):
        plot_valence_arousal_comparison(
            'DEAP',
            arousal_pub_deap, valence_pub_deap,
            arousal_priv_deap, valence_priv_deap,
            plots_dir, num_samples=10
        )
        plot_sampled_mean_points(
            'DEAP',
            arousal_pub_deap, valence_pub_deap,
            arousal_priv_deap, valence_priv_deap,
            plots_dir,
            num_groups=4, samples_per_group=38 # Changed to 4 groups of 38 samples
        )
    else:
        print("Insufficient data for DEAP. Ensure files exist in correct folders.")

    # --- Processing for ImagEEG (Graz) ---
    print("\n##### Processing for ImagEEG (Graz) #####")
    arousal_pub_graz = load_labels_graz('PUBLIC', 'arousal_pubblico')
    valence_pub_graz = load_labels_graz('PUBLIC', 'valence_pubblico')
    arousal_priv_graz = load_labels_graz('PRIVATE', 'rate_arousal_privato')
    valence_priv_graz = load_labels_graz('PRIVATE', 'rate_valence_privata')

    if all(arr.size > 0 for arr in [arousal_pub_graz, valence_pub_graz, arousal_priv_graz, valence_priv_graz]):
        plot_valence_arousal_comparison(
            'ImagEEG (Graz)',
            arousal_pub_graz, valence_pub_graz,
            arousal_priv_graz, valence_priv_graz,
            plots_dir, num_samples=10
        )
        plot_sampled_mean_points(
            'ImagEEG (Graz)',
            arousal_pub_graz, valence_pub_graz,
            arousal_priv_graz, valence_priv_graz,
            plots_dir,
            num_groups=4, samples_per_group=38 # Changed to 4 groups of 38 samples
        )
    else:
        print("Insufficient data for ImagEEG (Graz). Ensure files exist in correct folders.")

    print("\n\nProcessing complete.")