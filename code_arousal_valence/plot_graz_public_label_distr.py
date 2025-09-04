import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Import pandas for Excel export

def plot_emotion_distribution(emotion_type, all_summary_data):
    """
    Carica i file .npy delle label specificate (arousal_pubblico o valence_pubblico) dal dataset GRAZ,
    e visualizza diverse distribuzioni: originale, troncata, arrotondata,
    e le versioni binarizzate di troncata e arrotondata con soglia 5.
    Ogni grafico viene anche salvato nella cartella 'plots'.
    Vengono anche create e salvate immagini comparative.
    """
    file_pattern = f"{emotion_type}_pubblico"
    data_dir = os.path.join(".", "NPY_TRAINING_DISK_CACHE", "GRAZ_NPY", "LABEL", "PUBLIC")

    # Cartella dove salvare i plot
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True) # Crea la cartella 'plots' se non esiste

    all_labels = []
    files_found = 0

    print(f"\n--- Elaborazione per {emotion_type.upper()} ---")
    print(f"Ricerca file in corso nella cartella: {os.path.abspath(data_dir)}")

    if not os.path.isdir(data_dir):
        print(f"ERRORE: La cartella '{data_dir}' non è stata trovata.")
        print("Assicurati di eseguire lo script dalla cartella 'code_8' o che il percorso sia corretto.")
        return

    for filename in sorted(os.listdir(data_dir)):
        if file_pattern in filename and filename.endswith(".npy"):
            file_path = os.path.join(data_dir, filename)
            try:
                data = np.load(file_path)
                all_labels.append(data)
                files_found += 1
                print(f"  -> Caricato file: {filename} - Shape: {data.shape}")
            except Exception as e:
                print(f"Errore nel caricare il file {filename}: {e}")

    if not all_labels:
        print(f"\nNessun file '{file_pattern}.npy' trovato o caricato. Impossibile generare i grafici per {emotion_type}.")
        return

    concatenated_labels = np.concatenate(all_labels)
    num_samples = len(concatenated_labels)
    print(f"\n{files_found} file sono stati concatenati con successo per {emotion_type}.")
    print(f"Shape finale dei dati: {concatenated_labels.shape}")

    # Summary data for Excel
    summary_data = {
        'Tipo Emozione': emotion_type.capitalize(),
        'Trasformazione': [],
        'Min': [],
        'Max': [],
        'Media': [],
        'Numero Campioni': []
    }

    def add_to_summary(transform_name, data_array):
        summary_data['Trasformazione'].append(transform_name)
        summary_data['Min'].append(f"{data_array.min():.2f}")
        summary_data['Max'].append(f"{data_array.max():.2f}")
        summary_data['Media'].append(f"{data_array.mean():.2f}")
        summary_data['Numero Campioni'].append(len(data_array))

    # --- 1) Distribuzione delle label originali ---
    add_to_summary('Originale', concatenated_labels)
    print(f"Valori originali: Min={concatenated_labels.min():.2f}, Max={concatenated_labels.max():.2f}, Media={concatenated_labels.mean():.2f}")
    plt.figure(figsize=(10, 6))
    plt.hist(concatenated_labels, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'1) Distribuzione delle Label di {emotion_type.capitalize()} Originali', fontsize=16)
    plt.xlabel(f'Valore {emotion_type.capitalize()}', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_filename = os.path.join(plots_dir, f'{emotion_type}_1_original_distribution.png')
    plt.savefig(plot_filename)
    print(f"Grafico salvato: {plot_filename}")
    plt.show()

    # --- 2) Distribuzione delle label troncate all'intero ---
    truncated_labels = np.floor(concatenated_labels)
    add_to_summary('Troncata', truncated_labels)
    print(f"Valori troncati: Min={truncated_labels.min():.0f}, Max={truncated_labels.max():.0f}, Media={truncated_labels.mean():.2f}")
    plt.figure(figsize=(10, 6))
    min_val_trunc = int(truncated_labels.min())
    max_val_trunc = int(truncated_labels.max())
    bins_trunc = np.arange(min_val_trunc - 0.5, max_val_trunc + 1.5, 1)
    plt.hist(truncated_labels, bins=bins_trunc, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.title(f'2) Distribuzione delle Label di {emotion_type.capitalize()} Troncate all\'Intero', fontsize=16)
    plt.xlabel(f'Valore {emotion_type.capitalize()} Troncato', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.xticks(range(min_val_trunc, max_val_trunc + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_filename = os.path.join(plots_dir, f'{emotion_type}_2_truncated_distribution.png')
    plt.savefig(plot_filename)
    print(f"Grafico salvato: {plot_filename}")
    plt.show()

    # --- 3) Distribuzione delle label arrotondate all'intero più vicino ---
    rounded_labels = np.round(concatenated_labels)
    add_to_summary('Arrotondata', rounded_labels)
    print(f"Valori arrotondati: Min={rounded_labels.min():.0f}, Max={rounded_labels.max():.0f}, Media={rounded_labels.mean():.2f}")
    plt.figure(figsize=(10, 6))
    min_val_round = int(rounded_labels.min())
    max_val_round = int(rounded_labels.max())
    bins_round = np.arange(min_val_round - 0.5, max_val_round + 1.5, 1)
    plt.hist(rounded_labels, bins=bins_round, color='mediumseagreen', edgecolor='black', alpha=0.7)
    plt.title(f'3) Distribuzione delle Label di {emotion_type.capitalize()} Arrotondate all\'Intero più Vicino', fontsize=16)
    plt.xlabel(f'Valore {emotion_type.capitalize()} Arrotondato', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.xticks(range(min_val_round, max_val_round + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_filename = os.path.join(plots_dir, f'{emotion_type}_3_rounded_distribution.png')
    plt.savefig(plot_filename)
    print(f"Grafico salvato: {plot_filename}")
    plt.show()

    # --- 4) Distribuzione delle label troncate e binarizzate con soglia 5 ---
    binarized_truncated_labels = (truncated_labels >= 5).astype(int)
    counts_bin_trunc = np.bincount(binarized_truncated_labels)
    counts_bin_trunc_padded = np.pad(counts_bin_trunc, (0, max(0, 2 - len(counts_bin_trunc))), 'constant')
    add_to_summary('Troncata & Binarizzata (0/1)', np.array([0]*counts_bin_trunc_padded[0] + [1]*counts_bin_trunc_padded[1])) # Represent as 0/1 array for min/max/mean
    print(f"Valori troncati e binarizzati (Soglia 5): Conteggio 0={counts_bin_trunc_padded[0]}, Conteggio 1={counts_bin_trunc_padded[1]}")

    plt.figure(figsize=(8, 6))
    plt.bar(['< 5 (0)', '>= 5 (1)'], counts_bin_trunc_padded, color=['darkorange', 'purple'], edgecolor='black', alpha=0.7)
    plt.title(f'4) Distribuzione Binarizzata (Soglia 5) delle Label {emotion_type.capitalize()} Troncate', fontsize=16)
    plt.xlabel('Classe Binarizzata', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_filename = os.path.join(plots_dir, f'{emotion_type}_4_truncated_binarized_distribution.png')
    plt.savefig(plot_filename)
    print(f"Grafico salvato: {plot_filename}")
    plt.show()

    # --- 5) Distribuzione delle label arrotondate e binarizzate con soglia 5 ---
    binarized_rounded_labels = (rounded_labels >= 5).astype(int)
    counts_bin_round = np.bincount(binarized_rounded_labels)
    counts_bin_round_padded = np.pad(counts_bin_round, (0, max(0, 2 - len(counts_bin_round))), 'constant')
    add_to_summary('Arrotondata & Binarizzata (0/1)', np.array([0]*counts_bin_round_padded[0] + [1]*counts_bin_round_padded[1])) # Represent as 0/1 array for min/max/mean
    print(f"Valori arrotondati e binarizzati (Soglia 5): Conteggio 0={counts_bin_round_padded[0]}, Conteggio 1={counts_bin_round_padded[1]}")

    plt.figure(figsize=(8, 6))
    plt.bar(['< 5 (0)', '>= 5 (1)'], counts_bin_round_padded, color=['darkorange', 'purple'], edgecolor='black', alpha=0.7)
    plt.title(f'5) Distribuzione Binarizzata (Soglia 5) delle Label {emotion_type.capitalize()} Arrotondate', fontsize=16)
    plt.xlabel('Classe Binarizzata', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_filename = os.path.join(plots_dir, f'{emotion_type}_5_rounded_binarized_distribution.png')
    plt.savefig(plot_filename)
    print(f"Grafico salvato: {plot_filename}")
    plt.show()
    
    # Aggiungi i dati riassuntivi per questa emozione alla lista globale
    all_summary_data.append(pd.DataFrame(summary_data))

    # --- Immagini Comparative ---
    # Comparazione Troncata vs Arrotondata (Plot 2 vs Plot 3)
    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1) # 1 riga, 2 colonne, primo subplot
    plt.hist(truncated_labels, bins=bins_trunc, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.title(f'Troncata all\'Intero ({emotion_type.capitalize()})', fontsize=14)
    plt.xlabel(f'Valore {emotion_type.capitalize()} Troncato', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.xticks(range(min_val_trunc, max_val_trunc + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2) # 1 riga, 2 colonne, secondo subplot
    plt.hist(rounded_labels, bins=bins_round, color='mediumseagreen', edgecolor='black', alpha=0.7)
    plt.title(f'Arrotondata all\'Intero ({emotion_type.capitalize()})', fontsize=14)
    plt.xlabel(f'Valore {emotion_type.capitalize()} Arrotondato', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.xticks(range(min_val_round, max_val_round + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(f'Comparazione: Troncata vs Arrotondata per {emotion_type.capitalize()}', fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta layout per il suptitle
    plot_filename = os.path.join(plots_dir, f'{emotion_type}_comparative_truncated_vs_rounded.png')
    plt.savefig(plot_filename)
    print(f"Grafico comparativo salvato: {plot_filename}")
    plt.show()

    # Comparazione Binarizzata Troncata vs Binarizzata Arrotondata (Plot 4 vs Plot 5)
    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    plt.bar(['< 5 (0)', '>= 5 (1)'], counts_bin_trunc_padded, color=['darkorange', 'purple'], edgecolor='black', alpha=0.7)
    plt.title(f'Binarizzata (Soglia 5) Troncata ({emotion_type.capitalize()})', fontsize=14)
    plt.xlabel('Classe Binarizzata', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.bar(['< 5 (0)', '>= 5 (1)'], counts_bin_round_padded, color=['darkorange', 'purple'], edgecolor='black', alpha=0.7)
    plt.title(f'Binarizzata (Soglia 5) Arrotondata ({emotion_type.capitalize()})', fontsize=14)
    plt.xlabel('Classe Binarizzata', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(f'Comparazione: Binarizzata Troncata vs Arrotondata per {emotion_type.capitalize()}', fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(plots_dir, f'{emotion_type}_comparative_binarized_truncated_vs_rounded.png')
    plt.savefig(plot_filename)
    print(f"Grafico comparativo salvato: {plot_filename}")
    plt.show()


if __name__ == '__main__':
    all_summary_data = [] # Lista per raccogliere i DataFrame di riepilogo
    plot_emotion_distribution("arousal", all_summary_data)
    plot_emotion_distribution("valence", all_summary_data)

    # Concatena tutti i DataFrame di riepilogo in uno unico
    final_summary_df = pd.concat(all_summary_data, ignore_index=True)

    # Salva il DataFrame finale in un file Excel
    excel_filename = "summary_emotion_data.xlsx"
    excel_path = os.path.join("plots", excel_filename) # Salva l'Excel nella cartella 'plots'
    final_summary_df.to_excel(excel_path, index=False)
    print(f"\nFile Excel di riepilogo salvato: {excel_path}")