import os
import numpy as np
import pandas as pd # Importa la libreria pandas

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

if __name__ == '__main__':
    print("Inizio calcolo SMAPE e preparazione per il salvataggio in Excel...\n")

    results = {} # Dizionario per conservare i risultati SMAPE

    # --- Processing for DEAP ---
    print("##### Calcolo SMAPE per DEAP #####")
    arousal_pub_deap = load_labels_deap('PUBLIC', 'arousal_pubblica')
    valence_pub_deap = load_labels_deap('PUBLIC', 'valence_pubblica')
    arousal_priv_deap = load_labels_deap('PRIVATE', 'arousal_privata')
    valence_priv_deap = load_labels_deap('PRIVATE', 'valence_privata')

    if all(arr.size > 0 for arr in [arousal_pub_deap, valence_pub_deap, arousal_priv_deap, valence_priv_deap]):
        min_len_deap = min(len(arousal_pub_deap), len(valence_pub_deap), len(arousal_priv_deap), len(valence_priv_deap))
        
        arousal_pub_deap_trimmed = arousal_pub_deap[:min_len_deap]
        valence_pub_deap_trimmed = valence_pub_deap[:min_len_deap]
        arousal_priv_deap_trimmed = arousal_priv_deap[:min_len_deap]
        valence_priv_deap_trimmed = valence_priv_deap[:min_len_deap]

        smape_valence_deap = smape(valence_pub_deap_trimmed, valence_priv_deap_trimmed)
        smape_arousal_deap = smape(arousal_pub_deap_trimmed, arousal_priv_deap_trimmed)

        print(f"  SMAPE Valenza DEAP: {smape_valence_deap:.2f}%")
        print(f"  SMAPE Arousal DEAP: {smape_arousal_deap:.2f}%")
        
        results['DEAP'] = {
            'Valenza SMAPE': f"{smape_valence_deap:.2f}%",
            'Arousal SMAPE': f"{smape_arousal_deap:.2f}%"
        }
    else:
        print("  Dati insufficienti per DEAP per calcolare lo SMAPE.")
        results['DEAP'] = {
            'Valenza SMAPE': "Dati non disponibili",
            'Arousal SMAPE': "Dati non disponibili"
        }

    print("\n" + "-"*40 + "\n")

    # --- Processing for ImagEEG (Graz) ---
    print("##### Calcolo SMAPE per ImagEEG (Graz) #####")
    arousal_pub_graz = load_labels_graz('PUBLIC', 'arousal_pubblico')
    valence_pub_graz = load_labels_graz('PUBLIC', 'valence_pubblico')
    arousal_priv_graz = load_labels_graz('PRIVATE', 'rate_arousal_privato')
    valence_priv_graz = load_labels_graz('PRIVATE', 'rate_valence_privata')

    if all(arr.size > 0 for arr in [arousal_pub_graz, valence_pub_graz, arousal_priv_graz, valence_priv_graz]):
        min_len_graz = min(len(arousal_pub_graz), len(valence_pub_graz), len(arousal_priv_graz), len(valence_priv_graz))

        arousal_pub_graz_trimmed = arousal_pub_graz[:min_len_graz]
        valence_pub_graz_trimmed = valence_pub_graz[:min_len_graz]
        arousal_priv_graz_trimmed = arousal_priv_graz[:min_len_graz]
        valence_priv_graz_trimmed = valence_priv_graz[:min_len_graz]

        smape_valence_graz = smape(valence_pub_graz_trimmed, valence_priv_graz_trimmed)
        smape_arousal_graz = smape(arousal_pub_graz_trimmed, arousal_priv_graz_trimmed)

        print(f"  SMAPE Valenza ImagEEG (Graz): {smape_valence_graz:.2f}%")
        print(f"  SMAPE Arousal ImagEEG (Graz): {smape_arousal_graz:.2f}%")

        results['ImagEEG (Graz)'] = {
            'Valenza SMAPE': f"{smape_valence_graz:.2f}%",
            'Arousal SMAPE': f"{smape_arousal_graz:.2f}%"
        }
    else:
        print("  Dati insufficienti per ImagEEG (Graz) per calcolare lo SMAPE.")
        results['ImagEEG (Graz)'] = {
            'Valenza SMAPE': "Dati non disponibili",
            'Arousal SMAPE': "Dati non disponibili"
        }
    
    print("\n" + "-"*40 + "\n")

    # --- Salvataggio dei risultati in un file Excel ---
    output_df = pd.DataFrame.from_dict(results, orient='index')
    output_filename = "risultati_smape.xlsx"
    
    try:
        output_df.to_excel(output_filename, index_label="Dataset")
        print(f"Risultati SMAPE salvati in: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"ERRORE durante il salvataggio del file Excel: {e}")

    print("\nCalcolo SMAPE e salvataggio completato.")