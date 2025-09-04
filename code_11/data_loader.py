# data_loader.py
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from typing import Tuple, List, Dict, Any, Generator
import collections # Per _custom_undersample


def _custom_undersample(
    samples: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bilancia un dataset (campioni ed etichette) sottocampionando casualmente
    le classi più numerose per eguagliare la dimensione della classe meno numerosa.

    Args:
        samples (np.ndarray): Array NumPy contenente i campioni (features).
                              La prima dimensione deve corrispondere al numero di campioni.
        labels (np.ndarray): Array NumPy 1D contenente le etichette di classe
                             per ogni campione. Deve avere la stessa lunghezza
                             della prima dimensione di 'samples'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Una tupla contenente:
            - balanced_samples: Array NumPy con i campioni bilanciati.
            - balanced_labels: Array NumPy con le etichette corrispondenti.
            Entrambi avranno n_classi * dimensione_classe_minima campioni totali.
            L'ordine dei campioni/etichette restituiti è casuale.

    Raises:
        ValueError: Se 'samples' e 'labels' non hanno dimensioni compatibili
                    o se l'array delle etichette non è 1D o è vuoto.
    """
    if samples.shape[0] != labels.shape[0]:
        raise ValueError(
            f"La prima dimensione di 'samples' ({samples.shape[0]}) deve "
            f"corrispondere alla lunghezza di 'labels' ({labels.shape[0]})"
        )
    if labels.ndim != 1:
         raise ValueError(
             f"'labels' deve essere un array 1D, ma ha {labels.ndim} dimensioni."
         )
    if len(labels) == 0:
         raise ValueError("Gli input 'samples' e 'labels' non possono essere vuoti.")

    if samples.shape[0] == 0:
        print("Attenzione: 'samples' è vuoto. Nessun bilanciamento custom_undersampling possibile.")
        return samples, labels

    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)
    if n_classes < 2:
         print("Attenzione: Trovata meno di 2 classi. Nessun bilanciamento custom_undersampling necessario/possibile.")
         return samples, labels
    
    min_size = counts.min()
    print(f"--- Bilanciamento Classi (Custom Undersampling) --- Target size per classe: {min_size}")

    balanced_indices: List[int] = []
    for label_class in unique_labels:
        indices_class = np.where(labels == label_class)[0]
        indices_to_keep = np.random.choice(indices_class, size=min_size, replace=False)
        balanced_indices.extend(indices_to_keep.tolist())

    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    balanced_samples = samples[balanced_indices]
    balanced_labels = labels[balanced_indices]
    print(f"Campioni dopo custom_undersampling: {balanced_samples.shape[0]}")
    print("--- Bilanciamento Custom Undersampling Completato ---")
    return balanced_samples, balanced_labels

def _binarize_labels(labels: np.ndarray, threshold: Any) -> np.ndarray:
    """
    Binarizza le etichette.
    Se threshold è un float, binarizza un array 1D in 2 classi.
    Se threshold è una tupla/lista (val_thresh, aro_thresh) o un dizionario
    {'valence': val_thresh, 'arousal': aro_thresh}, binarizza un array 2D
    (valence, arousal) in 4 classi (HVHA, HVLA, LVHA, LVLA).

    Class Mapping for 4 classes:
    HVHA: 0 (Valence >= threshold_val, Arousal >= threshold_aro)
    HVLA: 1 (Valence >= threshold_val, Arousal <  threshold_aro)
    LVHA: 2 (Valence <  threshold_val, Arousal >= threshold_aro)
    LVLA: 3 (Valence <  threshold_val, Arousal <  threshold_aro)
    """
    if isinstance(threshold, (float, int)):
        # Comportamento esistente per binarizzazione a 2 classi
        if labels.ndim > 1:
            # Se per caso le labels 1D sono caricate come [[valore], [valore]], le appiattiamo
            if labels.shape[1] == 1:
                labels = labels.flatten()
            else:
                # Caso in cui sia un one-hot encoding o labels multidimensionali non previste
                print("Attenzione: _binarize_labels ha ricevuto label multidimensionali per soglia singola. Tentativo di usare argmax.")
                labels = np.argmax(labels, axis=1) # Se le label sono one-hot, prendi l'indice della classe
        labels_rounded = np.round(labels) # Label arrotondate all'intero più vicino
        return (labels_rounded >= threshold).astype(np.int_)
    
    # NUOVA CONDIZIONE: se threshold è un dizionario e contiene le chiavi 'valence' e 'arousal'
    elif isinstance(threshold, dict) and 'valence' in threshold and 'arousal' in threshold:
        val_thresh = threshold['valence']
        aro_thresh = threshold['arousal']
    # CONDIZIONE ESISTENTE: se threshold è una tupla/lista di lunghezza 2
    elif isinstance(threshold, (tuple, list)) and len(threshold) == 2:
        val_thresh, aro_thresh = threshold
    else:
        raise ValueError(f"Tipo di soglia di binarizzazione non supportato: {type(threshold)}. Deve essere float/int, una tupla/lista di 2 float/int, o un dizionario con chiavi 'valence' e 'arousal'.")
    
    # Logica comune per binarizzazione a 4 classi (raggiunta sia da tupla/lista che da dizionario)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(
            f"Per la binarizzazione a 4 classi, le 'labels' devono essere un array 2D con 2 colonne (Valence, Arousal), "
            f"ma la shape fornita è {labels.shape}."
        )
    
    valence_labels = np.round(labels[:, 0]) # Prima colonna è Valence
    arousal_labels = np.round(labels[:, 1]) # Seconda colonna è Arousal

    # Inizializza l'array delle nuove label a 4 classi
    binned_labels = np.zeros(labels.shape[0], dtype=np.int_)

    # HVHA: Valence >= val_thresh, Arousal >= aro_thresh -> Classe 0
    binned_labels[(valence_labels >= val_thresh) & (arousal_labels >= aro_thresh)] = 0
    
    # HVLA: Valence >= val_thresh, Arousal < aro_thresh -> Classe 1
    binned_labels[(valence_labels >= val_thresh) & (arousal_labels < aro_thresh)] = 1
    
    # LVHA: Valence < val_thresh, Arousal >= aro_thresh -> Classe 2
    binned_labels[(valence_labels < val_thresh) & (arousal_labels >= aro_thresh)] = 2
    
    # LVLA: Valence < val_thresh, Arousal < aro_thresh -> Classe 3
    binned_labels[(valence_labels < val_thresh) & (arousal_labels < aro_thresh)] = 3
    
    print(f"Binarizzazione a 4 classi (Valence >={val_thresh}, Arousal >={aro_thresh}) completata.")
    unique_binned, counts_binned = np.unique(binned_labels, return_counts=True)
    print(f"  Nuova distribuzione classi: {dict(zip(unique_binned, counts_binned))}")

    return binned_labels

def _load_all_subject_data_segmented_concatenated(config: dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]], List[str]]:
    """
    Carica i dati EEG e le etichette *GIÀ SEGMENTATI* da tutti i soggetti.
    Concatena i dati di tutti i soggetti in array globali.
    Restituisce i dati globali, una mappa degli indi    ci per soggetto, e gli ID dei soggetti validi.

    Args:
        config (dict): Dizionario di configurazione.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]], List[str]]:
        global_eeg_data, global_labels, subject_indices_map, valid_subject_ids

    """
    # I percorsi ora puntano ai dati SEGMENTATI (es. ./NPY_TRAINING_DISK_CACHE/DEAP_NPY/EEG)
    eeg_data_dir_segmented = config['eeg_data_dir_raw'] # This now points to segmented data
    labels_data_dir_base_segmented = config['labels_data_dir_base_raw'] # This now points to segmented data
    label_type_upper = config['label_type'].upper()
    label_metric = config['label_metric']
    labels_data_dir = os.path.join(labels_data_dir_base_segmented, label_type_upper)

    eeg_files_paths = sorted([os.path.join(eeg_data_dir_segmented, f) for f in os.listdir(eeg_data_dir_segmented) if f.endswith('_eeg.npy')])
    
    # Gestione speciale per la metrica "valence_arousal_4class"
    if label_metric == 'valence_arousal_4class':
        # Dobbiamo caricare sia valence che arousal
        valence_label_name = 'valence_pubblica' if label_type_upper == 'PUBLIC' else 'valence' if label_type_upper == 'PRIVATE' and 'DEAP_BDF' in config['eeg_data_dir_raw'] else 'valence_privata'
        arousal_label_name = 'arousal_pubblica' if label_type_upper == 'PUBLIC' else 'arousal' if label_type_upper == 'PRIVATE' and 'DEAP_BDF' in config['eeg_data_dir_raw'] else 'arousal_privata'
        
        # Per GRAZ le etichette private sono 'rate_valence_privata' e 'rate_arousal_privato'
        if 'GRAZ' in config['eeg_data_dir_raw'] and label_type_upper == 'PRIVATE':
            valence_label_name = 'rate_valence_privata'
            arousal_label_name = 'rate_arousal_privato'
        elif 'GRAZ' in config['eeg_data_dir_raw'] and label_type_upper == 'PUBLIC':
            valence_label_name = 'valence_pubblico'
            arousal_label_name = 'arousal_pubblico'

        valence_files_paths = sorted([os.path.join(labels_data_dir, f) for f in os.listdir(labels_data_dir) if f.endswith(f'_{valence_label_name}.npy')])
        arousal_files_paths = sorted([os.path.join(labels_data_dir, f) for f in os.listdir(labels_data_dir) if f.endswith(f'_{arousal_label_name}.npy')])

        label_files_paths_map = {}
        for vp_file in valence_files_paths:
            sub_id = os.path.basename(vp_file).replace(f'_{valence_label_name}.npy', '')
            if sub_id not in label_files_paths_map: label_files_paths_map[sub_id] = {}
            label_files_paths_map[sub_id]['valence_path'] = vp_file
        for ap_file in arousal_files_paths:
            sub_id = os.path.basename(ap_file).replace(f'_{arousal_label_name}.npy', '')
            if sub_id not in label_files_paths_map: label_files_paths_map[sub_id] = {}
            label_files_paths_map[sub_id]['arousal_path'] = ap_file

        # Ora creiamo la subject_map combinando EEG e Label (Valence + Arousal)
        subject_map = {}
        for eeg_file in eeg_files_paths:
            sub_id = os.path.basename(eeg_file).split('_')[0]
            if sub_id not in subject_map: subject_map[sub_id] = {}
            subject_map[sub_id]['eeg_path'] = eeg_file
        
        for sub_id, paths in label_files_paths_map.items():
            if sub_id in subject_map:
                subject_map[sub_id].update(paths)

        valid_subject_ids = sorted([
            s for s, files in subject_map.items()
            if 'eeg_path' in files and 'valence_path' in files and 'arousal_path' in files
        ])

    else:
        # Logica esistente per singole metriche (valence, arousal, happiness, etc.)
        # QUESTA LOGICA FUNZIONA ANCHE PER LE NUOVE LABEL HNS
        label_files_paths = sorted([
            os.path.join(labels_data_dir, f)
            for f in os.listdir(labels_data_dir)
            if f.endswith(f'_{label_metric}.npy')
        ])

        subject_map = {}
        for eeg_file in eeg_files_paths:
            sub_id = os.path.basename(eeg_file).split('_')[0]
            if sub_id not in subject_map: subject_map[sub_id] = {}
            subject_map[sub_id]['eeg_path'] = eeg_file

        for label_file in label_files_paths:
            sub_id = os.path.basename(label_file).replace(f'_{label_metric}.npy', '')
            if sub_id not in subject_map:
                subject_map[sub_id] = {}
            subject_map[sub_id]['label_path'] = label_file
        
        valid_subject_ids = sorted([
            s for s, files in subject_map.items()
            if 'eeg_path' in files and 'label_path' in files
        ])


    if not valid_subject_ids:
        raise ValueError(f"Nessun soggetto trovato con dati EEG e label segmentati corrispondenti per la metrica '{label_metric}'. Controllare i path: EEG dir '{eeg_data_dir_segmented}', Label dir '{labels_data_dir}'.")
    
    all_eeg_data_list, all_labels_data_list = [], []
    subject_indices_map = {}
    current_idx_start = 0

    for sub_id in valid_subject_ids:
        eeg = np.load(subject_map[sub_id]['eeg_path']).astype(np.float32)
        
        if label_metric == 'valence_arousal_4class':
            valence_labels = np.load(subject_map[sub_id]['valence_path']).astype(np.float32)
            arousal_labels = np.load(subject_map[sub_id]['arousal_path']).astype(np.float32)
            # Combina valence e arousal in un array 2D per ogni sample
            labels = np.stack((valence_labels, arousal_labels), axis=1) # Shape (num_samples, 2)
        else:
            labels = np.load(subject_map[sub_id]['label_path']).astype(np.float32)

        if eeg.shape[0] != labels.shape[0]:
            min_samples = min(eeg.shape[0], labels.shape[0])
            print(f"Attenzione: Mismatch nel numero di campioni tra EEG ({eeg.shape[0]}) e label ({labels.shape[0]}) per il soggetto {sub_id}. Verranno usati min({eeg.shape[0]}, {labels.shape[0]}) campioni.")
            eeg, labels = eeg[:min_samples], labels[:min_samples]
        
        if eeg.shape[0] == 0 or labels.shape[0] == 0:
            print(f"Attenzione: Dati EEG o label vuoti per il soggetto {sub_id}. Soggetto saltato.")
            continue

        # Se non è valence_arousal_4class, assicurati che labels sia 1D
        if label_metric != 'valence_arousal_4class' and labels.ndim > 1:
            labels = labels.squeeze()
            if labels.ndim > 1:
                 raise ValueError(f"Le label per il soggetto {sub_id} sono multidimensionali ({labels.shape}) anche dopo lo squeeze. Previste label 1D.")
                 
        all_eeg_data_list.append(eeg)
        all_labels_data_list.append(labels)
        num_samples_subject = eeg.shape[0]
        subject_indices_map[sub_id] = (current_idx_start, current_idx_start + num_samples_subject)
        current_idx_start += num_samples_subject

    if not all_eeg_data_list:
        raise ValueError("Nessun dato EEG/label valido caricato dopo aver iterato sui soggetti.")

    global_eeg_data = np.concatenate(all_eeg_data_list, axis=0)
    global_labels = np.concatenate(all_labels_data_list, axis=0)
    
    return global_eeg_data, global_labels, subject_indices_map, valid_subject_ids


def get_data_splits(config: dict, hpo_subset_ratio: float = 1.0) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]], None, None]:
    """
    Generatore che carica, segmenta e fornisce i dati (X, y) suddivisi per train, validation, e test
    in base allo scenario di training specificato nella configurazione.

    Scenari supportati: 'simple', 'kfold', 'loso'.

    Args:
        config (dict): Dizionario di configurazione.

    Yields:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, info_dict)
    """

    scenario = config['training_scenario']
    bin_thresh = config.get('binarized_threshold')
    balance_strat = config.get('balancing_strategy')
    label_metric = config['label_metric']
    
    print(f"\n--- Preparazione dati per scenario: {scenario} ---")
    
    # Caricamento dati già segmentati
    global_eeg_data_segmented, global_labels_segmented, subject_indices_map_segmented, valid_subject_ids = \
        _load_all_subject_data_segmented_concatenated(config)

    if global_eeg_data_segmented.shape[0] == 0:
        print("ATTENZIONE: Nessun dato EEG/label segmentato valido caricato. Impossibile procedere.")
        return

    print(f"Dati segmentati globali caricati: EEG shape {global_eeg_data_segmented.shape}, Labels shape {global_labels_segmented.shape}")
    print(f"Soggetti disponibili con dati segmentati: {len(valid_subject_ids)}")

    # --- APPLICAZIONE SUBSET PER HPO (se richiesto) ---
    if hpo_subset_ratio < 1.0:
        print(f"  Applicazione subset per HPO: {hpo_subset_ratio * 100:.2f}% dei dati totali.")
        
        labels_for_subset_stratification = global_labels_segmented.copy()
        # NUOVA MODIFICA: Non binarizzare le label HNS
        if bin_thresh is not None and label_metric not in ['hns_public', 'hns_private']:
            labels_for_subset_stratification = _binarize_labels(labels_for_subset_stratification, bin_thresh)
        
        can_stratify_subset = len(np.unique(labels_for_subset_stratification)) > 1 and \
                              len(labels_for_subset_stratification) * hpo_subset_ratio >= len(np.unique(labels_for_subset_stratification))
        
        if can_stratify_subset:
            global_eeg_data_segmented, _, global_labels_segmented, _ = train_test_split(
                global_eeg_data_segmented, global_labels_segmented,
                train_size=hpo_subset_ratio,
                stratify=labels_for_subset_stratification,
                random_state=config.get('global_shuffle_seed', 42)
            )
            print(f"  Dati dopo subset HPO (stratificato): EEG shape {global_eeg_data_segmented.shape}, Labels shape {global_labels_segmented.shape}")
        else:
            print(f"  ATTENZIONE: Impossibile stratificare il subset HPO. Eseguo subset non stratificato.")
            global_eeg_data_segmented, _, global_labels_segmented, _ = train_test_split(
                global_eeg_data_segmented, global_labels_segmented,
                train_size=hpo_subset_ratio,
                random_state=config.get('global_shuffle_seed', 42)
            )
            print(f"  Dati dopo subset HPO (non stratificato): EEG shape {global_eeg_data_segmented.shape}, Labels shape {global_labels_segmented.shape}")

    if scenario == 'k_simple':
        n_k_repetitions = config.get('k_simple_repetitions', config.get('kfold_splits', 1))
        print(f"\n  Scenario K-Simple: Ripetizioni K={n_k_repetitions}")

        for i_k_rep in range(n_k_repetitions):
            print(f"    K-Simple - Ripetizione {i_k_rep + 1}/{n_k_repetitions}")
            X_fold_concatenated = global_eeg_data_segmented
            y_fold_concatenated = global_labels_segmented
            
            # NUOVA MODIFICA: Non binarizzare le label HNS
            if bin_thresh is not None and label_metric not in ['hns_public', 'hns_private']:
                print(f"      [K-Simple Rip. {i_k_rep+1}] Binarizzazione globale con soglia: {bin_thresh}")
                y_fold_concatenated = _binarize_labels(y_fold_concatenated, bin_thresh)

            X_processed, y_processed = shuffle(X_fold_concatenated, y_fold_concatenated, random_state=i_k_rep)

            if balance_strat == 'custom_undersampling':
                print(f"      [K-Simple Rip. {i_k_rep+1}] Bilanciamento globale con custom_undersampling...")
                if X_processed.shape[0] > 0 and len(np.unique(y_processed)) >= 2:
                    X_processed, y_processed = _custom_undersample(X_processed, y_processed)
                else:
                    print(f"      [K-Simple Rip. {i_k_rep+1}] Bilanciamento saltato: dati insufficienti.")

            y_stratify_temp = y_processed
            can_stratify_split_temp = (len(np.unique(y_stratify_temp)) > 1 and len(y_stratify_temp) * 0.20 >= len(np.unique(y_stratify_temp)))
            
            if can_stratify_split_temp:
                X_temp, X_test_final, y_temp, y_test_final = train_test_split(X_processed, y_processed, test_size=0.20, random_state=i_k_rep, stratify=y_stratify_temp)
            else:
                print(f"      [K-Simple Rip. {i_k_rep+1}] Impossibile stratificare il primo split. Eseguo split non stratificato.")
                X_temp, X_test_final, y_temp, y_test_final = train_test_split(X_processed, y_processed, test_size=0.20, random_state=i_k_rep, shuffle=False)
            
            can_stratify_split_train_val = (len(np.unique(y_temp)) > 1 and len(y_temp) * 0.25 >= len(np.unique(y_temp)))
            
            if can_stratify_split_train_val:
                X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_temp, y_temp, test_size=0.25, random_state=i_k_rep, stratify=y_temp)
            else:
                print(f"      [K-Simple Rip. {i_k_rep+1}] Impossibile stratificare il secondo split. Eseguo split non stratificato.")
                X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_temp, y_temp, test_size=0.25, random_state=i_k_rep, shuffle=False)

            info = {'fold': i_k_rep + 1, 'total_folds': n_k_repetitions, 'scenario_info': f'k_simple repetition {i_k_rep + 1}'}
            yield X_train_final, y_train_final, X_val_final, y_val_final, X_test_final, y_test_final, info

    elif scenario == 'kfold':
        n_splits = config['kfold_splits']
        print(f"\n  Scenario K-Fold (Standard): Splits K={n_splits}")

        labels_for_kfold_stratification = global_labels_segmented.copy()
        # NUOVA MODIFICA: Non binarizzare le label HNS
        if bin_thresh is not None and label_metric not in ['hns_public', 'hns_private']:
            print(f"    [K-Fold Global] Binarizzazione globale con soglia: {bin_thresh}")
            labels_for_kfold_stratification = _binarize_labels(labels_for_kfold_stratification, bin_thresh)

        global_seed = config.get('global_shuffle_seed', 42)
        shuffled_eeg_segmented, shuffled_labels_for_kfold = shuffle(
            global_eeg_data_segmented, labels_for_kfold_stratification, random_state=global_seed
        )
        print(f"    [K-Fold Global] Shuffle globale applicato.")

        can_stratify_kfold = (len(np.unique(shuffled_labels_for_kfold)) > 1 and 
                              len(shuffled_labels_for_kfold) >= n_splits * len(np.unique(shuffled_labels_for_kfold)))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=False) if can_stratify_kfold else StratifiedKFold(n_splits=n_splits, shuffle=False)
        if not can_stratify_kfold:
            print(f"    ATTENZIONE: Impossibile stratificare KFold. Eseguo KFold non stratificato.")

        i_fold_count = 0
        for train_val_idx, test_idx in skf.split(shuffled_eeg_segmented, shuffled_labels_for_kfold):
            i_fold_count += 1
            print(f"\n    K-Fold - Fold {i_fold_count}/{n_splits}")

            eeg_train_val_fold_segmented = shuffled_eeg_segmented[train_val_idx]
            labels_train_val_fold_segmented = shuffled_labels_for_kfold[train_val_idx]
            eeg_test_fold_segmented = shuffled_eeg_segmented[test_idx]
            labels_test_fold_segmented = shuffled_labels_for_kfold[test_idx]

            val_split_percentage_of_train_fold = 0.20 
            can_stratify_val_split = (len(np.unique(labels_train_val_fold_segmented)) > 1 and 
                                      len(labels_train_val_fold_segmented) * val_split_percentage_of_train_fold >= len(np.unique(labels_train_val_fold_segmented)))
            
            if eeg_train_val_fold_segmented.shape[0] < 2:
                print(f"      ATTENZIONE Fold {i_fold_count}: Non abbastanza campioni per splittare. Usando tutto per train.")
                X_train_final, y_train_final = eeg_train_val_fold_segmented, labels_train_val_fold_segmented
                X_val_final, y_val_final = np.empty((0, *eeg_train_val_fold_segmented.shape[1:]), dtype=eeg_train_val_fold_segmented.dtype), np.empty((0,), dtype=labels_train_val_fold_segmented.dtype)
            elif can_stratify_val_split:
                X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(eeg_train_val_fold_segmented, labels_train_val_fold_segmented, test_size=val_split_percentage_of_train_fold, shuffle=True, stratify=labels_train_val_fold_segmented, random_state=i_fold_count)
            else:
                print(f"      ATTENZIONE Fold {i_fold_count}: Impossibile stratificare split train/val. Eseguo split non stratificato.")
                X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(eeg_train_val_fold_segmented, labels_train_val_fold_segmented, test_size=val_split_percentage_of_train_fold, shuffle=True, random_state=i_fold_count)

            X_test_final, y_test_final = eeg_test_fold_segmented, labels_test_fold_segmented

            if balance_strat == 'custom_undersampling':
                print(f"      [K-Fold {i_fold_count}] Bilanciamento del SOLO training set con custom_undersampling...")
                if X_train_final.shape[0] > 0 and len(np.unique(y_train_final)) >= 2:
                    X_train_final, y_train_final = _custom_undersample(X_train_final, y_train_final)
                else:
                    print(f"      [K-Fold {i_fold_count}] Bilanciamento saltato.")

            info = {'fold': i_fold_count, 'total_folds': n_splits, 'scenario_info': f'kfold fold {i_fold_count}'}
            yield X_train_final, y_train_final, X_val_final, y_val_final, X_test_final, y_test_final, info

    elif scenario == 'loso':
        num_val_subjects = config.get('loso_val_subjects', 2)
        limit = config.get('loso_test_limit')
        subjects_to_process_for_test = valid_subject_ids[:limit] if limit is not None else valid_subject_ids
        
        if limit is not None:
             print(f"  --- ESECUZIONE LOSO LIMITATA: {len(subjects_to_process_for_test)} soggetti come test set ---")

        for i_fold, test_sub_id in enumerate(subjects_to_process_for_test):
            print(f"\n  LOSO - Fold {i_fold + 1}/{len(subjects_to_process_for_test)} (Test Subject: {test_sub_id})")

            remaining_subs = [s for s in valid_subject_ids if s != test_sub_id]
            if len(remaining_subs) < num_val_subjects:
                print(f"    ATTENZIONE: Non abbastanza soggetti rimanenti. Saltando il fold per {test_sub_id}.")
                continue
            np.random.seed(i_fold)
            val_sub_ids = np.random.choice(remaining_subs, size=num_val_subjects, replace=False).tolist()
            train_sub_ids = [s for s in remaining_subs if s not in val_sub_ids]

            def get_segmented_data_for_subject_group(sub_ids_group):
                eeg_list, labels_list = [], []
                for sub_id_in_group in sub_ids_group:
                    start_idx, end_idx = subject_indices_map_segmented[sub_id_in_group]
                    eeg_list.append(global_eeg_data_segmented[start_idx:end_idx])
                    labels_list.append(global_labels_segmented[start_idx:end_idx])
                if not eeg_list:
                    dummy_eeg_shape = (0, *global_eeg_data_segmented.shape[1:])
                    return np.empty(dummy_eeg_shape, dtype=global_eeg_data_segmented.dtype), np.empty((0,), dtype=global_labels_segmented.dtype)
                return np.concatenate(eeg_list, axis=0), np.concatenate(labels_list, axis=0)

            X_train_final, y_train_final_orig = get_segmented_data_for_subject_group(train_sub_ids)
            X_val_final, y_val_final_orig = get_segmented_data_for_subject_group(val_sub_ids)
            X_test_final, y_test_final_orig = get_segmented_data_for_subject_group([test_sub_id])
            
            # NUOVA MODIFICA: Non binarizzare le label HNS
            if bin_thresh is not None and label_metric not in ['hns_public', 'hns_private']:
                print(f"    [LOSO Fold {i_fold+1}] Binarizzazione dei set con soglia: {bin_thresh}")
                y_train_final = _binarize_labels(y_train_final_orig, bin_thresh)
                y_val_final = _binarize_labels(y_val_final_orig, bin_thresh)
                y_test_final = _binarize_labels(y_test_final_orig, bin_thresh)
            else:
                y_train_final, y_val_final, y_test_final = y_train_final_orig, y_val_final_orig, y_test_final_orig

            X_train_final, y_train_final = shuffle(X_train_final, y_train_final, random_state=i_fold)
            X_val_final, y_val_final = shuffle(X_val_final, y_val_final, random_state=i_fold + 1)
            X_test_final, y_test_final = shuffle(X_test_final, y_test_final, random_state=i_fold + 2)
            
            if balance_strat == 'custom_undersampling':
                print(f"    [LOSO Fold {i_fold+1}] Bilanciamento del SOLO training set...")
                if X_train_final.shape[0] > 0 and len(np.unique(y_train_final)) >= 2:
                    X_train_final, y_train_final = _custom_undersample(X_train_final, y_train_final)
                else:
                    print(f"    [LOSO Fold {i_fold+1}] Bilanciamento saltato.")

            info = {'fold': i_fold + 1, 'total_folds': len(subjects_to_process_for_test), 'test_subject': test_sub_id, 'val_subjects': val_sub_ids, 'train_subjects': train_sub_ids}
            yield X_train_final, y_train_final, X_val_final, y_val_final, X_test_final, y_test_final, info

    elif scenario == 'simple':
        print("ATTENZIONE: Lo scenario 'simple' è obsoleto, uso 'k_simple'.")
        config['training_scenario'] = 'k_simple'
        config['k_simple_repetitions'] = 1
        yield from get_data_splits(config)

    else:
        raise ValueError(f"Scenario di training '{scenario}' non supportato.")