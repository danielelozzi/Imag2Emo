import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Verifica e installa le librerie se necessario
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Le librerie 'matplotlib' e 'seaborn' non sono installate. Installale con:")
    print("pip install matplotlib seaborn")
    exit()

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
    data = np.round(data, 0).astype(int)
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

def save_confusion_matrix(y_true, y_pred, title, filename, labels):
    """
    Generates and saves a confusion matrix plot as a PDF.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", cbar=True, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Public Labels')
    plt.xlabel('Private Labels')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    print("Inizio calcolo e preparazione dati per la tabella completa...\n")

    # Creazione della cartella per i report se non esiste
    output_dir = "classification_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        
        # Etichette per i report
        valence_labels = ['LV', 'HV']
        arousal_labels = ['LA', 'HA']

        # Salvataggio Classification Report e Confusion Matrix - Valenza
        print("\n### DEAP - Elaborazione Report Valenza ###")
        valence_report_deap = classification_report(valence_pub_deap_trimmed, valence_priv_deap_trimmed, output_dict=True, zero_division=0, target_names=valence_labels)
        df_valence_deap = pd.DataFrame(valence_report_deap).transpose()
        df_valence_deap.to_csv(os.path.join(output_dir, "deap_valence_report.csv"))
        print(f"Report Valenza DEAP salvato in: {os.path.abspath(os.path.join(output_dir, 'deap_valence_report.csv'))}")
        save_confusion_matrix(valence_pub_deap_trimmed, valence_priv_deap_trimmed, "Confusion Matrix DEAP Valence", os.path.join(output_dir, "deap_valence_cm.pdf"), valence_labels)
        print(f"Confusion Matrix Valenza DEAP salvata in: {os.path.abspath(os.path.join(output_dir, 'deap_valence_cm.pdf'))}")

        # Salvataggio Classification Report e Confusion Matrix - Arousal
        print("\n### DEAP - Elaborazione Report Arousal ###")
        arousal_report_deap = classification_report(arousal_pub_deap_trimmed, arousal_priv_deap_trimmed, output_dict=True, zero_division=0, target_names=arousal_labels)
        df_arousal_deap = pd.DataFrame(arousal_report_deap).transpose()
        df_arousal_deap.to_csv(os.path.join(output_dir, "deap_arousal_report.csv"))
        print(f"Report Arousal DEAP salvato in: {os.path.abspath(os.path.join(output_dir, 'deap_arousal_report.csv'))}")
        save_confusion_matrix(arousal_pub_deap_trimmed, arousal_priv_deap_trimmed, "Confusion Matrix DEAP Arousal", os.path.join(output_dir, "deap_arousal_cm.pdf"), arousal_labels)
        print(f"Confusion Matrix Arousal DEAP salvata in: {os.path.abspath(os.path.join(output_dir, 'deap_arousal_cm.pdf'))}")

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
        
        # Etichette per i report
        valence_labels = ['LV', 'HV']
        arousal_labels = ['LA', 'HA']

        # Salvataggio Classification Report e Confusion Matrix - Valenza
        print("\n### ImagEEG (Graz) - Elaborazione Report Valenza ###")
        valence_report_graz = classification_report(valence_pub_graz_trimmed, valence_priv_graz_trimmed, output_dict=True, zero_division=0, target_names=valence_labels)
        df_valence_graz = pd.DataFrame(valence_report_graz).transpose()
        df_valence_graz.to_csv(os.path.join(output_dir, "graz_valence_report.csv"))
        print(f"Report Valenza ImagEEG (Graz) salvato in: {os.path.abspath(os.path.join(output_dir, 'graz_valence_report.csv'))}")
        save_confusion_matrix(valence_pub_graz_trimmed, valence_priv_graz_trimmed, "Confusion Matrix ImagEEG Valence", os.path.join(output_dir, "graz_valence_cm.pdf"), valence_labels)
        print(f"Confusion Matrix Valenza ImagEEG (Graz) salvata in: {os.path.abspath(os.path.join(output_dir, 'graz_valence_cm.pdf'))}")

        # Salvataggio Classification Report e Confusion Matrix - Arousal
        print("\n### ImagEEG (Graz) - Elaborazione Report Arousal ###")
        arousal_report_graz = classification_report(arousal_pub_graz_trimmed, arousal_priv_graz_trimmed, output_dict=True, zero_division=0, target_names=arousal_labels)
        df_arousal_graz = pd.DataFrame(arousal_report_graz).transpose()
        df_arousal_graz.to_csv(os.path.join(output_dir, "graz_arousal_report.csv"))
        print(f"Report Arousal ImagEEG (Graz) salvato in: {os.path.abspath(os.path.join(output_dir, 'graz_arousal_report.csv'))}")
        save_confusion_matrix(arousal_pub_graz_trimmed, arousal_priv_graz_trimmed, "Confusion Matrix ImagEEG Arousal", os.path.join(output_dir, "graz_arousal_cm.pdf"), arousal_labels)
        print(f"Confusion Matrix Arousal ImagEEG (Graz) salvata in: {os.path.abspath(os.path.join(output_dir, 'graz_arousal_cm.pdf'))}")

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
    print("Output a terminale della tabella riassuntiva:")
    
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