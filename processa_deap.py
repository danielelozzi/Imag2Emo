# processa_deap.py

import os
import numpy as np
import pandas as pd
import mne # Importa MNE per leggere i file .set

def processa_e_salva_dati_deap(
    cartella_input_deap: str,
    cartella_output_base_deap: str,
    deap_participant_rating_path: str,
    deap_video_list_path: str
):
    """
    Processa i file .set del dataset DEAP, estrae i dati EEG e le etichette (private e pubbliche)
    dai file Excel forniti, e li salva come file .npy in una struttura di cartelle organizzata.
    Si assume che i file .set siano nominati come 'sXX_eeg_annotated.set' o simile,
    da cui si può estrare 'sXX' come subject_id.
    Le epoche EEG nei file .set sono attese essere nell'ordine dei trial.
    La frequenza di campionamento è 500 Hz e le epoche durano 60 secondi.

    Args:
        cartella_input_deap (str): Il percorso della cartella contenente i file .set.
        cartella_output_base_deap (str): Il percorso base dove salvare i file .npy.
        deap_participant_rating_path (str): Percorso al file 'participant_rating.xls'.
        deap_video_list_path (str): Percorso al file 'video_list.xls'.
    Raises:
        FileNotFoundError: Se la cartella di input o i file XLS non vengono trovati.
        Exception: Per errori durante il processamento o il salvataggio dei dati.
    """
    if not os.path.exists(cartella_input_deap):
        raise FileNotFoundError(f"ERRORE CRITICO: La cartella di input DEAP '{cartella_input_deap}' non è stata trovata.")
    if not os.path.exists(deap_participant_rating_path):
        raise FileNotFoundError(f"ERRORE CRITICO: Il file '{deap_participant_rating_path}' non è stato trovato.")
    if not os.path.exists(deap_video_list_path):
        raise FileNotFoundError(f"ERRORE CRITICO: Il file '{deap_video_list_path}' non è stato trovato.")

    try:
        df_ratings_all = pd.read_excel(deap_participant_rating_path)
        df_videos_all = pd.read_excel(deap_video_list_path)
    except Exception as e:
        print(f"ERRORE CRITICO durante il caricamento dei file Excel: {e}")
        raise

    # Definizione della struttura delle cartelle
    eeg_output_dir = os.path.join(cartella_output_base_deap, "EEG")
    label_output_dir_private = os.path.join(cartella_output_base_deap, "LABEL", "PRIVATE")
    label_output_dir_public = os.path.join(cartella_output_base_deap, "LABEL", "PUBLIC")

    os.makedirs(eeg_output_dir, exist_ok=True)
    os.makedirs(label_output_dir_private, exist_ok=True)
    os.makedirs(label_output_dir_public, exist_ok=True)

    print(f"Inizio elaborazione dei file .set nella cartella: '{cartella_input_deap}'")

    file_list = [f for f in os.listdir(cartella_input_deap) if f.endswith(".set") and f.startswith("s")]
    if not file_list:
        print(f"\nERRORE CRITICO: Nessun file corrispondente al pattern 's*.set' trovato in '{cartella_input_deap}'.")
        print("Verificare che il percorso sia corretto e che i file siano presenti. Elaborazione DEAP interrotta.")
        return

    print(f"Trovati {len(file_list)} file .set da analizzare.")
    print(f"I file .npy verranno salvati in: '{cartella_output_base_deap}'")

    processed_subject_count = 0
    for nome_file_set in file_list:
        if nome_file_set.endswith(".set") and nome_file_set.startswith("s"):
            percorso_file_set = os.path.join(cartella_input_deap, nome_file_set)
            subject_id_str = nome_file_set.split('_')[0]
            participant_num = int(subject_id_str[1:])

            print(f"\n  Processando soggetto: {subject_id_str} (Participant ID: {participant_num}) da file: {nome_file_set}")

            try:
                print(f"    Caricamento dati EEG da {nome_file_set}...")
                raw = mne.io.read_epochs_eeglab(percorso_file_set)
                
                # --- FIX APPLICATO QUI ---
                # Rimuoviamo .transpose(2, 0, 1) perché raw.get_data() 
                # restituisce già la forma corretta: (trials, channels, timepoints).
                eeg_data = raw.get_data() 
                print(f"    Dati EEG caricati: {eeg_data.shape}, sfreq: {raw.info['sfreq']}")
                
                num_epochs_eeg = eeg_data.shape[0]
                if num_epochs_eeg == 0:
                    print(f"    ATTENZIONE: Nessuna epoca EEG trovata nel file {nome_file_set} per {subject_id_str}. Saltando.")
                    continue

                subject_ratings = df_ratings_all[df_ratings_all['Participant_id'] == participant_num].copy()
                if subject_ratings.empty:
                    print(f"    ATTENZIONE: Nessun rating trovato per Participant_id {participant_num} ({subject_id_str}) in '{deap_participant_rating_path}'. Saltando.")
                    continue

                subject_ratings = subject_ratings.sort_values(by='Trial').reset_index(drop=True)
                
                num_epochs_labels = len(subject_ratings)
                min_epochs = min(num_epochs_eeg, num_epochs_labels)
                if num_epochs_eeg != num_epochs_labels:
                    print(f"    ATTENZIONE: Mismatch nel numero di epoche per {subject_id_str}. EEG: {num_epochs_eeg}, Labels: {num_epochs_labels}. Verranno usate min({num_epochs_eeg}, {num_epochs_labels}) epoche.")
                    eeg_data = eeg_data[:min_epochs]
                    subject_ratings = subject_ratings.iloc[:min_epochs]
                
                if min_epochs == 0 :
                     print(f"    ATTENZIONE: Numero minimo di epoche è 0 per {subject_id_str} dopo il check. Saltando.")
                     continue

                valence_privata = subject_ratings['Valence'].values.astype(np.float32)
                arousal_privata = subject_ratings['Arousal'].values.astype(np.float32)

                experiment_ids = subject_ratings['Experiment_id'].values
                temp_df_exp_ids = pd.DataFrame({'Experiment_id': experiment_ids})
                merged_public_labels = pd.merge(temp_df_exp_ids, df_videos_all[['Experiment_id', 'AVG_Valence', 'AVG_Arousal']], on='Experiment_id', how='left', sort=False)

                valence_pubblica = merged_public_labels['AVG_Valence'].values.astype(np.float32)
                arousal_pubblica = merged_public_labels['AVG_Arousal'].values.astype(np.float32)

                if np.isnan(valence_pubblica).any() or np.isnan(arousal_pubblica).any():
                    print(f"    ATTENZIONE: Trovati NaN nelle etichette pubbliche per {subject_id_str} dopo il merge.")

                eeg_filename = f"{subject_id_str}_eeg.npy"
                np.save(os.path.join(eeg_output_dir, eeg_filename), eeg_data)
                print(f"    - Salvato {eeg_filename} (Shape: {eeg_data.shape})")

                labels_to_save = {
                    'valence_privata': (valence_privata, label_output_dir_private),
                    'arousal_privata': (arousal_privata, label_output_dir_private),
                    'valence_pubblica': (valence_pubblica, label_output_dir_public),
                    'arousal_pubblica': (arousal_pubblica, label_output_dir_public),
                }

                for label_name, (data_array, out_dir) in labels_to_save.items():
                    label_filename_npy = f"{subject_id_str}_{label_name}.npy"
                    np.save(os.path.join(out_dir, label_filename_npy), data_array)
                    print(f"    - Salvato {label_filename_npy} (Shape: {data_array.shape})")
                
                processed_subject_count +=1

            except FileNotFoundError as e_fnf:
                print(f"    ERRORE File Non Trovato per {subject_id_str}: {e_fnf}. Saltando soggetto.")
            except Exception as e:
                print(f"    ERRORE durante l'elaborazione del soggetto {subject_id_str} (file: {nome_file_set}): {e}.")
                import traceback
                traceback.print_exc()

    if processed_subject_count == 0:
        print("\nATTENZIONE: Nessun soggetto DEAP è stato processato con successo. Controllare i log e i percorsi dei file.")
    else:
        print(f"\nElaborazione DEAP completata. Processati {processed_subject_count} soggetti.")