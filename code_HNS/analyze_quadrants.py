import os
import pandas as pd

def load_graz_data(folder_path="./label_private_filtered_p008_plus"):
    """
    Carica tutti i file CSV da una cartella specificata, li concatena e 
    restituisce un DataFrame pandas con le colonne di valenza e arousal 
    pubbliche e private.

    Args:
        folder_path (str): Il percorso della cartella contenente i file CSV.

    Returns:
        pd.DataFrame: Un DataFrame contenente le colonne:
                      'public_valence', 'public_arousal',
                      'private_valence', 'private_arousal'.
    
    Raises:
        FileNotFoundError: Se la cartella specificata non esiste o è vuota.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"La cartella specificata non è stata trovata: '{os.path.abspath(folder_path)}'")

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"Nessun file CSV trovato nella cartella: '{os.path.abspath(folder_path)}'")

    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            df_list.append(pd.read_csv(file_path))
        except Exception as e:
            print(f"Errore nel leggere il file {file}: {e}")
    
    if not df_list:
        raise ValueError("Nessun dato è stato caricato. Controlla i file CSV.")

    combined_df = pd.concat(df_list, ignore_index=True)

    column_mapping = {
        'valence_pubblico': 'public_valence',
        'arousal_pubblico': 'public_arousal',
        'rate_valence_privata': 'private_valence',
        'rate_arousal_privato': 'private_arousal'
    }

    required_columns = list(column_mapping.keys())
    if not all(col in combined_df.columns for col in required_columns):
        raise ValueError("Una o più colonne richieste non sono state trovate nei file CSV.")
        
    final_df = combined_df[required_columns].copy()
    final_df.rename(columns=column_mapping, inplace=True)
    
    print(f"Dati caricati e uniti da {len(csv_files)} file CSV. Totale campioni: {len(final_df)}.")
    return final_df

def get_quadrant(valence, arousal, center_x=5, center_y=5):
    """
    Restituisce il quadrante per una data coppia di valenza e arousal.
    Quadranti:
    1: Alta Valenza, Alto Arousal (HVHA)
    2: Bassa Valenza, Alto Arousal (LVHA)
    3: Bassa Valenza, Basso Arousal (LVLA)
    4: Alta Valenza, Basso Arousal (HVLA)
    """
    if valence >= center_x and arousal >= center_y:
        return 1 # HVHA
    elif valence < center_x and arousal >= center_y:
        return 2 # LVHA
    elif valence < center_x and arousal < center_y:
        return 3 # LVLA
    elif valence >= center_x and arousal < center_y:
        return 4 # HVLA

def analyze_quadrant_distribution(df: pd.DataFrame):
    """
    Analizza e stampa la distribuzione dei campioni nei quadranti per le label
    pubbliche e private.
    """
    df['public_quadrant'] = df.apply(lambda row: get_quadrant(row['public_valence'], row['public_arousal'], center_x=5, center_y=5), axis=1)
    df['private_quadrant'] = df.apply(lambda row: get_quadrant(row['private_valence'], row['private_arousal'], center_x=5, center_y=5), axis=1)

    print("\n--- Analisi Distribuzione nei Quadranti (Centro a V=5, A=5) ---")
    print("Q1: HVHA | Q2: LVHA | Q3: LVLA | Q4: HVLA\n")

    public_counts = df['public_quadrant'].value_counts().sort_index()
    print("Distribuzione LABEL PUBBLICHE:")
    print(public_counts)
    print("\nDistribuzione LABEL PRIVATE:")
    private_counts = df['private_quadrant'].value_counts().sort_index()
    print(private_counts)

if __name__ == "__main__":
    graz_df = load_graz_data(folder_path="../label_private_filtered_p008_plus")
    analyze_quadrant_distribution(graz_df)