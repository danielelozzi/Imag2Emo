# processa_deap_bdf.py
import os
import pickle
import numpy as np
import pandas as pd # Aggiungi l'import di pandas

def carica_dati_pickle_folder(cartella: str) -> dict:
    """
    Carica i file .dat contenenti dati pickle da una cartella specificata.
    ... (funzione invariata)
    """
    dati_cartella = {}
    if not os.path.exists(cartella):
        raise FileNotFoundError(f"Errore: la cartella '{cartella}' non è stata trovata.")

    for nome_file in os.listdir(cartella):
        if nome_file.endswith(".dat"):
            percorso_file = os.path.join(cartella, nome_file)
            try:
                with open(percorso_file, 'rb') as file:
                    dati = pickle.load(file, encoding='iso-8859-1')
                dati_cartella[nome_file] = dati
            except UnicodeDecodeError as e:
                print(f"ATTENZIONE: Errore di codifica nel file '{nome_file}': {e}. Questo file verrà saltato.")
            except pickle.UnpicklingError as e:
                print(f"ATTENZIONE: Errore di unpickling nel file '{nome_file}': {e}. Questo file verrà saltato.")
            except Exception as e:
                print(f"ATTENZIONE: Errore generico con il file '{nome_file}': {e}. Questo file verrà saltato.")
    return dati_cartella

def get_deap_public_labels(
    participant_rating_path: str,
    video_list_path: str,
    subject_id: str, # es. 's01'
    num_samples: int = 40 # Il numero fisso di sample per soggetto (40 video)
) -> dict:
    """
    Carica e prepara le label pubbliche (Valence e Arousal medie) per un soggetto DEAP.
    Le label vengono ordinate in base all'Experiment_id.

    Args:
        participant_rating_path (str): Percorso al file 'participant_rating.xls'.
        video_list_path (str): Percorso al file 'video_list.xls'.
        subject_id (str): L'ID del soggetto corrente (es. 's01').
        num_samples (int): Il numero di epoche/video per soggetto (default 40).

    Returns:
        dict: Un dizionario con le label pubbliche 'valence_pubblica' e 'arousal_pubblica',
              ordinate per Experiment_id. I valori sono array numpy.
    Raises:
        FileNotFoundError: Se i file XLS non vengono trovati.
        ValueError: Se i dati nel file non sono come attesi o il soggetto non è trovato.
    """
    if not os.path.exists(participant_rating_path):
        raise FileNotFoundError(f"Errore: Il file '{participant_rating_path}' non è stato trovato.")
    if not os.path.exists(video_list_path):
        raise FileNotFoundError(f"Errore: Il file '{video_list_path}' non è stato trovato.")

    try:
        df_ratings = pd.read_excel(participant_rating_path)
        df_videos = pd.read_excel(video_list_path)

        # Filtra i rating per il soggetto corrente
        # Converti subject_id da 's01' a 1, 's10' a 10, etc.
        participant_num = int(subject_id[1:])
        subject_ratings = df_ratings[df_ratings['Participant_id'] == participant_num]

        if subject_ratings.empty:
            raise ValueError(f"Nessun rating trovato per il soggetto '{subject_id}' nel file '{participant_rating_path}'.")

        # Verifica che ci siano 40 rating per il soggetto
        if len(subject_ratings) != num_samples:
            print(f"ATTENZIONE: Trovati {len(subject_ratings)} rating per il soggetto {subject_id}, attesi {num_samples}.")

        # Unisci i rating del soggetto con la lista dei video basandoti su 'Experiment_id'
        # Assicurati che l'ordine sia basato sull'Experiment_id
        # Experiment_id va da 1 a 40 e corrisponde all'ordine dei video presentati.
        # Quindi dobbiamo assicurare che le label pubbliche siano in questo stesso ordine.
        # La colonna 'Experiment_id' in df_ratings per il soggetto, e in df_videos_sorted
        # corrisponde alla sequenza dei 40 campioni del soggetto.
        merged_data = pd.merge(subject_ratings, df_videos, on='Experiment_id', how='left')

        # Ordina per Experiment_id per garantire che l'ordine corrisponda ai sample EEG
        merged_data = merged_data.sort_values(by='Experiment_id').reset_index(drop=True)

        # Estrai le label pubbliche
        # La colonna 'AVG_Valence' dal file video_list.xls contiene la valence pubblica
        # La colonna 'AVG_Arousal' dal file video_list.xls contiene l'arousal pubblica
        public_valence = merged_data['AVG_Valence'].values
        public_arousal = merged_data['AVG_Arousal'].values

        if len(public_valence) != num_samples or len(public_arousal) != num_samples:
             raise ValueError(f"Il numero di label pubbliche non corrisponde ai {num_samples} campioni attesi per il soggetto {subject_id}.")


        # Verifica che non ci siano NaN (nel caso di merge non riuscito)
        if np.isnan(public_valence).any() or np.isnan(public_arousal).any():
            print(f"ATTENZIONE: Trovati valori NaN nelle label pubbliche per il soggetto {subject_id}. Controllare i file XLS.")
            # Potresti voler riempire con un valore medio o sollevare un errore qui.
            # Per ora, proseguiamo, ma potrebbe indicare un problema nei dati.


        return {
            'valence_pubblica': public_valence.astype(np.float32),
            'arousal_pubblica': public_arousal.astype(np.float32) # Assicurati il tipo float per compatibilità
        }

    except Exception as e:
        raise Exception(f"Errore durante il recupero delle label pubbliche DEAP per il soggetto '{subject_id}': {e}")


def processa_e_salva_dati_deap_bdf( # Renamed function
    cartella_input_deap: str,
    cartella_output_base_deap: str,
    deap_participant_rating_path: str, # Nuovo parametro
    deap_video_list_path: str # Nuovo parametro
):
    """
    Processa i file .dat del dataset DEAP, estrae i dati EEG e le etichette (private e pubbliche),
    e li salva come file .npy in una struttura di cartelle organizzata.

    Args:
        cartella_input_deap (str): Il percorso della cartella contenente i file .dat.
        cartella_output_base_deap (str): Il percorso base dove salvare i file .npy.
        deap_participant_rating_path (str): Percorso al file 'participant_rating.xls'.
        deap_video_list_path (str): Percorso al file 'video_list.xls'.
    Raises:
        FileNotFoundError: Se la cartella di input o i file XLS non vengono trovati.
        Exception: Per errori durante il processamento o il salvataggio dei dati.
    """
    try:
        dati_cartella = carica_dati_pickle_folder(cartella_input_deap)
    except FileNotFoundError as e:
        print(f"ERRORE CRITICO: Impossibile procedere. {e}")
        raise
    except Exception as e:
        print(f"ERRORE CRITICO durante il caricamento iniziale dei dati: {e}")
        raise

    # Definizione delle etichette private e della struttura delle cartelle
    private_labels_names = ['valence', 'arousal'] # Rimosso dominance e liking
    eeg_output_dir = os.path.join(cartella_output_base_deap, "EEG")
    label_output_dir_private = os.path.join(cartella_output_base_deap, "LABEL", "PRIVATE")
    label_output_dir_public = os.path.join(cartella_output_base_deap, "LABEL", "PUBLIC") # Nuova cartella per label pubbliche

    # Crea le directory se non esistono
    os.makedirs(eeg_output_dir, exist_ok=True)
    os.makedirs(label_output_dir_private, exist_ok=True)
    os.makedirs(label_output_dir_public, exist_ok=True) # Crea la nuova directory

    print(f"Inizio elaborazione dei file .dat nella cartella: '{cartella_input_deap}'")
    print(f"I file .npy verranno salvati in: '{cartella_output_base_deap}'")

    for nome_file, dati in dati_cartella.items():
        try:
            subject_id = nome_file.split('.')[0] # es. 's01', 's02', etc.

            # --- Estrai e salva i dati EEG ---
            eeg_data = dati['data']  # Shape: 40 x 40 x 8064 (sample x channel x timepoint)
            eeg_filename = f"{subject_id}_eeg.npy"
            eeg_filepath = os.path.join(eeg_output_dir, eeg_filename)
            np.save(eeg_filepath, eeg_data)
            print(f"  - Salvato {eeg_filename}")

            # --- Estrai e salva le label private ---
            labels_private = dati['labels']  # Shape: sample x label
            for i, label_name in enumerate(private_labels_names):
                single_label_data = labels_private[:, i]
                label_filename = f"{subject_id}_{label_name}.npy"
                label_filepath = os.path.join(label_output_dir_private, label_filename)
                np.save(label_filepath, single_label_data)
                print(f"  - Salvato {label_filename} (private)")

            # --- Estrai e salva le label pubbliche ---
            # Chiama la nuova funzione per ottenere le label pubbliche ordinate
            public_labels_data = get_deap_public_labels(
                deap_participant_rating_path,
                deap_video_list_path,
                subject_id
            )

            for label_name, label_array in public_labels_data.items():
                label_filename = f"{subject_id}_{label_name}.npy"
                label_filepath = os.path.join(label_output_dir_public, label_filename)
                np.save(label_filepath, label_array)
                print(f"  - Salvato {label_filename} (public)")

            print(f"  -- Completato elaborazione per {nome_file}\n")

        except KeyError as e:
            print(f"ERRORE: Chiave '{e}' non trovata nel dizionario per il file '{nome_file}'. Questo file verrà saltato.")
        except Exception as e:
            print(f"ERRORE: Errore imprevisto durante l'elaborazione del file '{nome_file}': {e}. Questo file verrà saltato.")

    print("\nElaborazione completata. Tutti i file .npy sono stati salvati (se non ci sono stati errori critici).")

# Questo blocco non verrà eseguito se il modulo viene importato
if __name__ == '__main__':
    print("Questo script è progettato per essere chiamato da un modulo orchestratore.")
    print("Eseguirlo direttamente potrebbe non avere l'effetto desiderato.")