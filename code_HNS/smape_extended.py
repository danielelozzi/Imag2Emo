import os
import numpy as np
import pandas as pd

# --- Funzioni load_labels dai file originali per DEAP e Graz ---
def load_labels_deap(label_type: str, label_metric: str) -> np.ndarray:
    """
    Loads and concatenates .npy files for a specific label metric from the DEAP dataset.
    """
    data_dir = os.path.join(".", "LABEL_DEAP", label_type.upper())
    
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
    data_dir = os.path.join(".", "LABEL_GRAZ", label_type.upper())
    
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

def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    smape_per_point = np.where(denominator == 0, 0, numerator / denominator)
    
    return np.mean(smape_per_point) * 100

def binarize(data, threshold=5):
    """
    Binarizes the data into 0s and 1s based on a threshold.
    """
    data = np.round(data, 0).astype(float)  # Ensure data is float for comparison
    return (data >= threshold).astype(int)

def count_labels(data):
    """
    Counts the occurrences of 0s and 1s in a binarized numpy array.
    """
    if data.size == 0:
        return {'0': 0, '1': 0}
    unique, counts = np.unique(data, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    return { '0': counts_dict.get(0, 0), '1': counts_dict.get(1, 0) }

if __name__ == '__main__':
    print("ATTENZIONE: SMAPE non è una metrica adatta per dati binari. I calcoli saranno eseguiti come richiesto, ma valuta l'uso di metriche di classificazione come l'Accuratezza.\n")
    print("Inizio calcolo e preparazione dati per la tabella completa...\n")

    results_data = {}

    # --- Processing for DEAP ---
    print("##### Dati per DEAP #####")
    arousal_pub_deap_orig = load_labels_deap('PUBLIC', 'arousal_pubblica')
    valence_pub_deap_orig = load_labels_deap('PUBLIC', 'valence_pubblica')
    arousal_priv_deap_orig = load_labels_deap('PRIVATE', 'arousal_privata')
    valence_priv_deap_orig = load_labels_deap('PRIVATE', 'valence_privata')

    if all(arr.size > 0 for arr in [arousal_pub_deap_orig, valence_pub_deap_orig, arousal_priv_deap_orig, valence_priv_deap_orig]):
        # Binarizzazione
        arousal_pub_deap = binarize(arousal_pub_deap_orig)
        valence_pub_deap = binarize(valence_pub_deap_orig)
        arousal_priv_deap = binarize(arousal_priv_deap_orig)
        valence_priv_deap = binarize(valence_priv_deap_orig)
        
        # Trimming
        min_len_deap = min(len(arousal_pub_deap), len(valence_pub_deap), len(arousal_priv_deap), len(valence_priv_deap))
        arousal_pub_deap_trimmed = arousal_pub_deap[:min_len_deap]
        valence_pub_deap_trimmed = valence_pub_deap[:min_len_deap]
        arousal_priv_deap_trimmed = arousal_priv_deap[:min_len_deap]
        valence_priv_deap_trimmed = valence_priv_deap[:min_len_deap]

        # Calcolo SMAPE
        smape_valence_deap = smape(valence_pub_deap_trimmed, valence_priv_deap_trimmed)
        smape_arousal_deap = smape(arousal_pub_deap_trimmed, arousal_priv_deap_trimmed)
        
        results_data['DEAP'] = {
            'Valence Pubblica (Iniziale)': len(valence_pub_deap_orig),
            'Arousal Pubblica (Iniziale)': len(arousal_pub_deap_orig),
            'Valence Privata (Iniziale)': len(valence_priv_deap_orig),
            'Arousal Privata (Iniziale)': len(arousal_priv_deap_orig),
            'Valence Pubblica (0/1)': str(count_labels(valence_pub_deap_trimmed)),
            'Arousal Pubblica (0/1)': str(count_labels(arousal_pub_deap_trimmed)),
            'Valence Privata (0/1)': str(count_labels(valence_priv_deap_trimmed)),
            'Arousal Privata (0/1)': str(count_labels(arousal_priv_deap_trimmed)),
            'SMAPE Valenza': f"{smape_valence_deap:.2f}%",
            'SMAPE Arousal': f"{smape_arousal_deap:.2f}%"
        }
    else:
        results_data['DEAP'] = {
            'Valence Pubblica (Iniziale)': "N/A",
            'Arousal Pubblica (Iniziale)': "N/A",
            'Valence Privata (Iniziale)': "N/A",
            'Arousal Privata (Iniziale)': "N/A",
            'Valence Pubblica (0/1)': "N/A",
            'Arousal Pubblica (0/1)': "N/A",
            'Valence Privata (0/1)': "N/A",
            'Arousal Privata (0/1)': "N/A",
            'SMAPE Valenza': "Dati non disponibili",
            'SMAPE Arousal': "Dati non disponibili"
        }

    print("\n" + "-"*40 + "\n")

    # --- Processing for ImagEEG (Graz) ---
    print("##### Dati per ImagEEG (Graz) #####")
    arousal_pub_graz_orig = load_labels_graz('PUBLIC', 'arousal_pubblico')
    valence_pub_graz_orig = load_labels_graz('PUBLIC', 'valence_pubblico')
    arousal_priv_graz_orig = load_labels_graz('PRIVATE', 'rate_arousal_privato')
    valence_priv_graz_orig = load_labels_graz('PRIVATE', 'rate_valence_privata')

    if all(arr.size > 0 for arr in [arousal_pub_graz_orig, valence_pub_graz_orig, arousal_priv_graz_orig, valence_priv_graz_orig]):
        # Binarizzazione
        arousal_pub_graz = binarize(arousal_pub_graz_orig)
        valence_pub_graz = binarize(valence_pub_graz_orig)
        arousal_priv_graz = binarize(arousal_priv_graz_orig)
        valence_priv_graz = binarize(valence_priv_graz_orig)

        # Trimming
        min_len_graz = min(len(arousal_pub_graz), len(valence_pub_graz), len(arousal_priv_graz), len(valence_priv_graz))
        arousal_pub_graz_trimmed = arousal_pub_graz[:min_len_graz]
        valence_pub_graz_trimmed = valence_pub_graz[:min_len_graz]
        arousal_priv_graz_trimmed = arousal_priv_graz[:min_len_graz]
        valence_priv_graz_trimmed = valence_priv_graz[:min_len_graz]
        
        # Calcolo SMAPE
        smape_valence_graz = smape(valence_pub_graz_trimmed, valence_priv_graz_trimmed)
        smape_arousal_graz = smape(arousal_pub_graz_trimmed, arousal_priv_graz_trimmed)

        results_data['ImagEEG (Graz)'] = {
            'Valence Pubblica (Iniziale)': len(valence_pub_graz_orig),
            'Arousal Pubblica (Iniziale)': len(arousal_pub_graz_orig),
            'Valence Privata (Iniziale)': len(valence_priv_graz_orig),
            'Arousal Privata (Iniziale)': len(valence_priv_graz_orig),
            'Valence Pubblica (0/1)': str(count_labels(valence_pub_graz_trimmed)),
            'Arousal Pubblica (0/1)': str(count_labels(arousal_pub_graz_trimmed)),
            'Valence Privata (0/1)': str(count_labels(valence_priv_graz_trimmed)),
            'Arousal Privata (0/1)': str(count_labels(arousal_priv_graz_trimmed)),
            'SMAPE Valenza': f"{smape_valence_graz:.2f}%",
            'SMAPE Arousal': f"{smape_arousal_graz:.2f}%"
        }
    else:
        results_data['ImagEEG (Graz)'] = {
            'Valence Pubblica (Iniziale)': "N/A",
            'Arousal Pubblica (Iniziale)': "N/A",
            'Valence Privata (Iniziale)': "N/A",
            'Arousal Privata (Iniziale)': "N/A",
            'Valence Pubblica (0/1)': "N/A",
            'Arousal Pubblica (0/1)': "N/A",
            'Valence Privata (0/1)': "N/A",
            'Arousal Privata (0/1)': "N/A",
            'SMAPE Valenza': "Dati non disponibili",
            'SMAPE Arousal': "Dati non disponibili"
        }
    
    print("\n" + "-"*40 + "\n")
    print("Output a terminale della tabella completa:")
    
    # Crea un dataframe dai dati raccolti
    output_df = pd.DataFrame.from_dict(results_data, orient='index')
    # Modifica l'ordine delle colonne per una migliore leggibilità
    column_order = [
        'Valence Pubblica (Iniziale)', 'Valence Privata (Iniziale)', 
        'Arousal Pubblica (Iniziale)', 'Arousal Privata (Iniziale)', 
        'Valence Pubblica (0/1)', 'Valence Privata (0/1)', 
        'Arousal Pubblica (0/1)', 'Arousal Privata (0/1)', 
        'SMAPE Valenza', 'SMAPE Arousal'
    ]
    output_df = output_df.reindex(columns=column_order)
    
    print(output_df)

    # --- Salvataggio dei risultati in un file Excel ---
    output_filename = "risultati_completi.xlsx"
    try:
        output_df.to_excel(output_filename, index_label="Dataset")
        print(f"\nRisultati completi salvati in: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"ERRORE durante il salvataggio del file Excel: {e}")
    
    # --- Generazione del codice LaTeX per la tabella completa ---
    latex_output_filename = "risultati_completi.tex"
    try:
        latex_code = output_df.to_latex(
            caption="Risultati completi: numerosità iniziali e SMAPE binarizzato",
            label="tab:risultati_completi",
            header=True,
            index=True,
            float_format="%.2f",
            na_rep="N/A",
            escape=False
        )
        with open(latex_output_filename, "w") as f:
            f.write(latex_code)
        print(f"Codice LaTeX della tabella completa salvato in: {os.path.abspath(latex_output_filename)}")
    except Exception as e:
        print(f"ERRORE durante il salvataggio del file LaTeX: {e}")

    print("\nCalcolo e salvataggio completato.")