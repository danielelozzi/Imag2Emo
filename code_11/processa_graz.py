# processa_graz.py
import os
import numpy as np
import pandas as pd
import mne # Assicurati di aver installato mne (pip install mne)

def get_similar_images_to_exclude(similar_pairs_csv_path: str) -> list:
    """
    Carica il file similar_image_pairs.csv e restituisce una lista dei nomi
    delle immagini da escludere (colonna 1).

    Args:
        similar_pairs_csv_path (str): Percorso al file CSV delle coppie simili.

    Returns:
        list: Una lista di stringhe, dove ogni stringa è il nome di un'immagine da escludere.
    Raises:
        FileNotFoundError: Se il file CSV non viene trovato.
        Exception: Per altri errori di lettura del CSV.
    """
    if not os.path.exists(similar_pairs_csv_path):
        raise FileNotFoundError(f"Errore: Il file '{similar_pairs_csv_path}' non è stato trovato.")
    try:
        # Assumiamo che la prima colonna contenga i nomi delle immagini da escludere
        df = pd.read_csv(similar_pairs_csv_path, header=None) # header=None se non ci sono intestazioni
        # Prendiamo solo la prima colonna e la convertiamo in lista di stringhe
        # Rimuoviamo l'estensione del file per facilitare il matching (es. .jpg, .png)
        images_to_exclude = [os.path.splitext(os.path.basename(img_path))[0] for img_path in df.iloc[:, 1].astype(str).tolist()]
        print(f"Caricate {len(images_to_exclude)} immagini da escludere da '{similar_pairs_csv_path}'.")
        return images_to_exclude
    except Exception as e:
        raise Exception(f"Errore durante la lettura del file '{similar_pairs_csv_path}': {e}")

def get_labels_and_map_epochs(
    subject_labels_csv_path: str,
    images_to_exclude: list
) -> tuple[pd.DataFrame, list]:
    """
    Carica il file CSV delle label per un soggetto, filtra le epoche da escludere
    e prepara i dati delle label e gli indici delle epoche valide.

    Args:
        subject_labels_csv_path (str): Percorso al file CSV delle label del soggetto.
        images_to_exclude (list): Lista dei nomi delle immagini da escludere.

    Returns:
        tuple[pd.DataFrame, list]: Un tuple contenente:
            - pd.DataFrame: DataFrame delle label filtrate e ordinate.
            - list: Una lista di indici delle epoche (samples) da mantenere dopo l'esclusione.
    Raises:
        FileNotFoundError: Se il file CSV delle label non viene trovato.
        ValueError: Se le colonne necessarie non sono presenti nel CSV.
        Exception: Per altri errori durante il caricamento o il processamento delle label.
    """
    if not os.path.exists(subject_labels_csv_path):
        raise FileNotFoundError(f"Errore: Il file delle label '{subject_labels_csv_path}' non è stato trovato.")

    try:
        df_labels = pd.read_csv(subject_labels_csv_path)

        required_cols = [
            'file_name', 'n_progressivo_pubblico', 'n_progressivo_privato',
            'label_emozione_pubblica', 'arousal_pubblico', 'valence_pubblico',
            'rate_happiness', 'rate_sadness', 'rate_fear', 'rate_surprise',
            'rate_anger', 'rate_disgust', 'rate_valence_privata', 'rate_arousal_privato'
        ]
        if not all(col in df_labels.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_labels.columns]
            raise ValueError(f"Colonne mancanti nel CSV delle label '{subject_labels_csv_path}': {missing_cols}")

        epochs_to_keep_mask = np.array([True] * len(df_labels))

        excluded_count = 0
        # Aggiungi un set per tenere traccia dei nomi delle immagini scartate per evitare duplicati nel log
        # Questo è utile se la stessa immagine è presente più volte nel CSV delle label
        logged_excluded_images = set()

        for i, row in df_labels.iterrows():
            image_name_from_label_csv = os.path.splitext(os.path.basename(row['file_name']))[0]
            if image_name_from_label_csv in images_to_exclude:
                epochs_to_keep_mask[i] = False
                excluded_count += 1
                if image_name_from_label_csv not in logged_excluded_images:
                    # Stampa l'informazione sull'epoca scartata
                    print(f"  - Scartata epoca {i} (immagine: '{image_name_from_label_csv}') dal file label: '{os.path.basename(subject_labels_csv_path)}'")
                    logged_excluded_images.add(image_name_from_label_csv)

        print(f"  - Epoche iniziali: {len(df_labels)}, Epoche scartate: {excluded_count}")

        df_filtered_labels = df_labels[epochs_to_keep_mask].copy()
        # L'ordinamento specifico per tipo di label (pubblico/privato)
        # verrà gestito nella funzione processa_e_salva_dati_graz.
        valid_epoch_indices = np.where(epochs_to_keep_mask)[0].tolist()

        return df_filtered_labels, valid_epoch_indices

    except Exception as e:
        raise Exception(f"Errore durante il processamento delle label per '{subject_labels_csv_path}': {e}")


def processa_e_salva_dati_graz(
    cartella_input_graz: str,
    cartella_labels_csv_graz: str,
    similar_pairs_csv_path: str,
    cartella_output_base_graz: str
):
    """
    Processa i file .set del dataset GRAZ, estrae i dati EEG e le etichette,
    escludendo le epoche specificate e li salva come file .npy.

    Args:
        cartella_input_graz (str): Percorso della cartella contenente i file .set.
        cartella_labels_csv_graz (str): Percorso della cartella contenente i file CSV delle label.
        similar_pairs_csv_path (str): Percorso al file CSV delle coppie di immagini simili da escludere.
        cartella_output_base_graz (str): Percorso base dove salvare i file .npy (e.g., "./NPY/GRAZ").
    Raises:
        FileNotFoundError: Se una delle cartelle/file di input non viene trovata.
        Exception: Per altri errori durante il processamento o il salvataggio dei dati.
    """
    print(f"\nInizio elaborazione dati GRAZ da: '{cartella_input_graz}'")
    print(f"I file .npy verranno salvati in: '{cartella_output_base_graz}'")

    # Carica la lista delle immagini da escludere una sola volta
    try:
        images_to_exclude = get_similar_images_to_exclude(similar_pairs_csv_path)
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile caricare le immagini da escludere. {e}")
        raise # Rilancia l'eccezione per la funzione chiamante

    # Definizione della struttura delle cartelle di output
    eeg_output_dir = os.path.join(cartella_output_base_graz, "EEG")
    label_output_dir_public = os.path.join(cartella_output_base_graz, "LABEL", "PUBLIC")
    label_output_dir_private = os.path.join(cartella_output_base_graz, "LABEL", "PRIVATE")

    # Crea le directory se non esistono
    os.makedirs(eeg_output_dir, exist_ok=True)
    os.makedirs(label_output_dir_public, exist_ok=True)
    os.makedirs(label_output_dir_private, exist_ok=True)

    # Definizione CORRETTA delle etichette pubbliche e private
    public_labels = ['label_emozione_pubblica', 'arousal_pubblico', 'valence_pubblico']
    private_labels = ['rate_happiness', 'rate_sadness', 'rate_fear', 'rate_surprise',
                      'rate_anger', 'rate_disgust', 'rate_valence_privata', 'rate_arousal_privato']
    
    # --- NUOVA MODIFICA: Mappatura per le classi HNS ---
    hns_map = {'H': 0, 'N': 1, 'S': 2}


    # Cerca i file .set per i soggetti P008 a P027
    subject_ids = [f"P{i:03d}" for i in range(8, 28)] # Genera P008, P009, ..., P027

    for subject_id in subject_ids:
        eeg_file_name = f"{subject_id}_prep.set"
        eeg_file_path = os.path.join(cartella_input_graz, eeg_file_name)

        # Il nome del file CSV delle label deve corrispondere al formato "label_PXX.csv"
        # Se i tuoi CSV si chiamano semplicemente PXX.csv, assicurati di aver aggiornato
        # il parametro GRAZ_LABELS_CSV_PATH in main_orchestrator.py
        # oppure cambia la riga seguente a f"{subject_id}.csv"
        label_csv_name = f"label_{subject_id}.csv"
        label_csv_path = os.path.join(cartella_labels_csv_graz, label_csv_name)


        print(f"\nProcessamento soggetto: {subject_id}")

        if not os.path.exists(eeg_file_path):
            print(f"ATTENZIONE: File EEG '{eeg_file_path}' non trovato per il soggetto {subject_id}. Saltando.")
            continue
        if not os.path.exists(label_csv_path):
            print(f"ATTENZIONE: File label CSV '{label_csv_path}' non trovato per il soggetto {subject_id}. Saltando.")
            continue

        try:
            # 1. Carica e filtra le label CSV
            df_filtered_labels, valid_epoch_indices = get_labels_and_map_epochs(
                label_csv_path, images_to_exclude
            )
            if df_filtered_labels.empty:
                print(f"  - Nessuna epoca valida rimasta per il soggetto {subject_id} dopo il filtraggio. Saltando.")
                continue

            # 2. Carica i dati EEG
            epochs_raw = mne.io.read_epochs_eeglab(
                eeg_file_path,
                verbose=False,
            )
            print(f"  - Caricate {len(epochs_raw)} epoche EEG raw da '{eeg_file_name}'.")

            # Estrai solo le epoche valide dal MNE Epochs object
            eeg_data_filtered = epochs_raw[valid_epoch_indices].get_data() # (samples, channels, times)
            print(f"  - Dopo filtraggio: {eeg_data_filtered.shape[0]} epoche EEG valide.")

            # Salva i dati EEG
            eeg_filename = f"{subject_id}_eeg.npy"
            eeg_filepath = os.path.join(eeg_output_dir, eeg_filename)
            np.save(eeg_filepath, eeg_data_filtered)
            print(f"  - Salvato {eeg_filename} in {eeg_output_dir}")

            # --- NUOVA MODIFICA: Generazione e Salvataggio Label HNS ---
            # Riordina per il progressivo pubblico per tutte le label pubbliche
            df_labels_sorted_public = df_filtered_labels.sort_values(by='n_progressivo_pubblico').reset_index(drop=True)
            
            # Label HNS pubblica
            public_hns_labels = df_labels_sorted_public['label_emozione_pubblica'].map(hns_map).values
            label_filename_hns_pub = f"{subject_id}_hns_public.npy"
            label_filepath_hns_pub = os.path.join(label_output_dir_public, label_filename_hns_pub)
            np.save(label_filepath_hns_pub, public_hns_labels)
            print(f"  - Salvato {label_filename_hns_pub} in {label_output_dir_public}")

            # Riordina per il progressivo privato per tutte le label private
            df_labels_sorted_private = df_filtered_labels.sort_values(by='n_progressivo_privato').reset_index(drop=True)

            # Label HNS privata (derivata)
            diff = df_labels_sorted_private['rate_happiness'] - df_labels_sorted_private['rate_sadness']
            conditions = [diff >= 2, diff <= -2]
            choices = [hns_map['H'], hns_map['S']]
            private_hns_labels = np.select(conditions, choices, default=hns_map['N'])
            label_filename_hns_priv = f"{subject_id}_hns_private.npy"
            label_filepath_hns_priv = os.path.join(label_output_dir_private, label_filename_hns_priv)
            np.save(label_filepath_hns_priv, private_hns_labels)
            print(f"  - Salvato {label_filename_hns_priv} in {label_output_dir_private}")
            # --- FINE NUOVA MODIFICA HNS ---

            # Salva le altre label pubbliche (usando il df già ordinato)
            for label_name in public_labels:
                label_data = df_labels_sorted_public[label_name].values
                # Se la label è quella categorica, mappala (già fatto per hns_public, ma per consistenza)
                if label_name == 'label_emozione_pubblica':
                    label_data = df_labels_sorted_public[label_name].map(hns_map).values
                label_filename = f"{subject_id}_{label_name}.npy"
                label_filepath = os.path.join(label_output_dir_public, label_filename)
                np.save(label_filepath, label_data)
                print(f"  - Salvato {label_filename} in {label_output_dir_public}")

            # Salva le altre label private (usando il df già ordinato)
            for label_name in private_labels:
                label_data = df_labels_sorted_private[label_name].values
                label_filename = f"{subject_id}_{label_name}.npy"
                label_filepath = os.path.join(label_output_dir_private, label_filename)
                np.save(label_filepath, label_data)
                print(f"  - Salvato {label_filename} in {label_output_dir_private}")

        except Exception as e:
            print(f"ERRORE durante il processamento del soggetto {subject_id}: {e}. Passando al prossimo soggetto.")
            continue # Passa al prossimo soggetto in caso di errore specifico

    print("\nElaborazione dati GRAZ completata.")

# Questo blocco non verrà eseguito se il modulo viene importato
if __name__ == '__main__':
    print("Questo script è progettato per essere chiamato da un modulo orchestratore.")
    print("Eseguirlo direttamente potrebbe non avere l'effetto desiderato.")