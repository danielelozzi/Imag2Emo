# prepare_training_segments.py
import os
import numpy as np
import warnings
import mne
from scipy.signal import resample
from numpy.typing import NDArray

def resample_data(X: NDArray, old_sfreq: int, new_sfreq: int) -> NDArray:
    """
    Ricampiona i dati EEG da una frequenza di campionamento a un'altra.
    """
    if X.shape[0] == 0:
        return X
    
    n_timepoints_old = X.shape[2]
    n_timepoints_new = int(n_timepoints_old * new_sfreq / old_sfreq)
    
    X_resampled = resample(X, n_timepoints_new, axis=2)
    
    return X_resampled.astype(np.float32)

def filter_data_butterworth(X: NDArray, sfreq: int, l_freq: float, h_freq: float, order: int) -> NDArray:
    """
    Applica un filtro passa-banda di Butterworth ai dati EEG.
    """
    if X.shape[0] == 0:
        return X
    
    nyquist_freq = sfreq / 2.0
    if h_freq is not None and h_freq >= nyquist_freq:
        warnings.warn(f"La frequenza di taglio superiore (h_freq={h_freq} Hz) è >= alla frequenza di Nyquist ({nyquist_freq} Hz). Il filtro passa-basso (superiore) verrà disabilitato (h_freq impostato a None).")
        h_freq = None

    if l_freq is not None and l_freq <= 0:
        raise ValueError(f"La frequenza di taglio inferiore (l_freq={l_freq} Hz) deve essere > 0.")

    X_for_filtering = X.astype(np.float64)
    X_filtered = mne.filter.filter_data(X_for_filtering, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, method='iir', iir_params={'order': order, 'ftype': 'butter', 'output': 'sos'}, verbose=False, copy=True)
    return X_filtered.astype(np.float32)

def generate_training_segments(
    processed_data_base_paths: dict,
    training_output_base_path: str,
    slicing_configs: dict,
    dataset_specific_info: dict,
    pipeline_config: dict
):
    """
    Genera segmenti di training (finestre) dai dati EEG pre-processati.
    """
    print("\n--- Inizio generazione segmenti di training ---")

    apply_resample_global = pipeline_config.get('apply_resample', False)
    new_sampling_rate_global = pipeline_config.get('new_sampling_rate')
    apply_butter_filter_global = pipeline_config.get('apply_butter_filter', False)
    butter_l_freq_global = pipeline_config.get('butter_l_freq')
    butter_h_freq_global = pipeline_config.get('butter_h_freq')
    butter_order_global = pipeline_config.get('butter_order')

    for dataset_name, p_data_base_path in processed_data_base_paths.items():
        print(f"\n  Processando dataset: {dataset_name}")
        
        dataset_upper = dataset_name.upper()
        slicing_config_orig = slicing_configs[dataset_upper]
        specific_info = dataset_specific_info[dataset_upper]
        subject_ids = specific_info['subject_ids']
        original_sfreq = specific_info['sampling_rate']

        current_sfreq = original_sfreq
        if apply_resample_global:
            current_sfreq = new_sampling_rate_global
            print(f"    Ricampionamento abilitato: {original_sfreq} Hz -> {current_sfreq} Hz per {dataset_name}.")
        
        slicing_config_for_segmentation = slicing_config_orig.copy()
        if apply_resample_global and original_sfreq != current_sfreq:
            scale_factor = current_sfreq / original_sfreq
            slicing_config_for_segmentation['start_time_idx'] = int(slicing_config_orig['start_time_idx'] * scale_factor)
            slicing_config_for_segmentation['end_time_idx'] = int(slicing_config_orig['end_time_idx'] * scale_factor)
            slicing_config_for_segmentation['sample_length_tp'] = int(slicing_config_orig['sample_length_tp'] * scale_factor)
            print(f"    Slicing config adjusted for {dataset_name}: {slicing_config_for_segmentation}")

        output_eeg_dir = os.path.join(training_output_base_path, f"{dataset_upper}_NPY", "EEG")
        output_label_base_dir = os.path.join(training_output_base_path, f"{dataset_upper}_NPY", "LABEL")

        os.makedirs(output_eeg_dir, exist_ok=True)
        os.makedirs(os.path.join(output_label_base_dir, "PUBLIC"), exist_ok=True)
        os.makedirs(os.path.join(output_label_base_dir, "PRIVATE"), exist_ok=True)

        for subject_id in subject_ids:
            print(f"    Processando soggetto: {subject_id}")

            eeg_file_name_orig = f"{subject_id}_eeg.npy"
            eeg_file_path_orig = os.path.join(p_data_base_path, "EEG", eeg_file_name_orig)

            if not os.path.exists(eeg_file_path_orig):
                print(f"      ATTENZIONE: File EEG '{eeg_file_path_orig}' non trovato per {subject_id}. Saltando soggetto.")
                continue
            
            try:
                eeg_data_orig = np.load(eeg_file_path_orig)
                print(f"      Caricato EEG per {subject_id}. Shape iniziale: {eeg_data_orig.shape}")
                if eeg_data_orig.shape[0] == 0:
                    print(f"      ATTENZIONE: Dati EEG caricati per {subject_id} sono vuoti (0 trial). Saltando soggetto.")
                    continue
            except Exception as e:
                print(f"      ERRORE nel caricare EEG per {subject_id}: {e}. Saltando soggetto.")
                continue

            if apply_resample_global and original_sfreq != current_sfreq:
                print(f"      Ricampionamento dati EEG per {subject_id} da {original_sfreq} Hz a {current_sfreq} Hz.")
                eeg_data_orig = resample_data(eeg_data_orig, original_sfreq, current_sfreq)
                print(f"      Nuova shape EEG dopo ricampionamento: {eeg_data_orig.shape}")

            if apply_butter_filter_global:
                print(f"      Applicazione filtro Butterworth per {subject_id} ({butter_l_freq_global}-{butter_h_freq_global} Hz, ordine {butter_order_global}).")
                eeg_data_orig = filter_data_butterworth(eeg_data_orig, current_sfreq, 
                                                        butter_l_freq_global, butter_h_freq_global, 
                                                        butter_order_global)
                print(f"      Shape EEG dopo filtraggio: {eeg_data_orig.shape}")

            # --- MODIFICA CHIAVE: APPLICA SELEZIONE CANALI QUI ---
            channel_indices = slicing_config_for_segmentation.get('channel_indices')
            if channel_indices is not None:
                print(f"      Selezione di {len(channel_indices)} canali specificati.")
                # La shape è (n_trials, n_channels, n_timepoints)
                eeg_data_orig = eeg_data_orig[:, channel_indices, :]
                print(f"      Nuova shape EEG dopo selezione canali: {eeg_data_orig.shape}")
            # --- FINE MODIFICA ---

            subject_segmented_eeg_list = []
            print(f"      Configurazione slicing per {subject_id}: start_idx={slicing_config_for_segmentation['start_time_idx']}, "
                  f"end_idx={slicing_config_for_segmentation['end_time_idx']}, segment_len_tp={slicing_config_for_segmentation['sample_length_tp']}")
            
            subject_segmented_labels_map = {}
            original_labels_for_subject = {}
            has_any_label = False
            for label_type_lower in ['public', 'private']:
                label_type_upper = label_type_lower.upper()
                original_labels_for_subject[label_type_upper] = {}
                for label_metric in specific_info[f'{label_type_lower}_labels']:
                    label_file_name_orig = f"{subject_id}_{label_metric}.npy"
                    label_file_path_orig = os.path.join(p_data_base_path, "LABEL", label_type_upper, label_file_name_orig)
                    if os.path.exists(label_file_path_orig):
                        try:
                            original_labels_for_subject[label_type_upper][label_metric] = np.load(label_file_path_orig)
                            has_any_label = True
                        except Exception as e:
                            print(f"      ATTENZIONE: Errore nel caricare label '{label_metric}' per {subject_id}: {e}.")
                    else:
                        print(f"      ATTENZIONE: File label '{label_file_path_orig}' non trovato.")
            
            if not has_any_label and eeg_data_orig.shape[0] > 0:
                print(f"      ATTENZIONE: Nessuna label trovata per {subject_id} ma dati EEG presenti.")

            start_idx = slicing_config_for_segmentation['start_time_idx']
            end_idx = slicing_config_for_segmentation['end_time_idx']
            segment_len_tp = slicing_config_for_segmentation['sample_length_tp']

            total_segments_for_subject = 0
            for trial_idx in range(eeg_data_orig.shape[0]):
                eeg_trial_full = eeg_data_orig[trial_idx]
                eeg_trial_sliced = eeg_trial_full[:, start_idx:end_idx]
                
                num_segments_created_for_this_trial = 0
                for s_start in range(0, eeg_trial_sliced.shape[1] - segment_len_tp + 1, segment_len_tp):
                    eeg_segment = eeg_trial_sliced[:, s_start : s_start + segment_len_tp]
                    if eeg_segment.shape[1] == segment_len_tp:
                        subject_segmented_eeg_list.append(eeg_segment)
                        num_segments_created_for_this_trial += 1
                total_segments_for_subject += num_segments_created_for_this_trial
                
                if num_segments_created_for_this_trial > 0:
                    for label_type_upper, metrics_data in original_labels_for_subject.items():
                        if label_type_upper not in subject_segmented_labels_map: subject_segmented_labels_map[label_type_upper] = {}
                        for label_metric, labels_array_orig in metrics_data.items():
                            if label_metric not in subject_segmented_labels_map[label_type_upper]: subject_segmented_labels_map[label_type_upper][label_metric] = []
                            
                            if trial_idx < len(labels_array_orig):
                                original_label_for_trial = labels_array_orig[trial_idx]
                                subject_segmented_labels_map[label_type_upper][label_metric].extend([original_label_for_trial] * num_segments_created_for_this_trial)
                            else:
                                print(f"      ATTENZIONE: Mismatch tra trial EEG e label {label_metric} per soggetto {subject_id} al trial {trial_idx}.")

            print(f"      Totale segmenti EEG generati per {subject_id}: {total_segments_for_subject}")

            if subject_segmented_eeg_list:
                final_eeg_segments_arr = np.array(subject_segmented_eeg_list).astype(np.float32)
                np.save(os.path.join(output_eeg_dir, f"{subject_id}_eeg.npy"), final_eeg_segments_arr)
                print(f"      Salvati EEG segmentati per {subject_id}: {final_eeg_segments_arr.shape}")

                for label_type_upper, metrics_data in subject_segmented_labels_map.items():
                    for label_metric, segmented_labels_list in metrics_data.items():
                        if segmented_labels_list:
                            final_label_segments_arr = np.array(segmented_labels_list).astype(np.float32)
                            output_label_dir = os.path.join(output_label_base_dir, label_type_upper)
                            np.save(os.path.join(output_label_dir, f"{subject_id}_{label_metric}.npy"), final_label_segments_arr)
                            print(f"        Salvate label segmentate {label_metric} ({label_type_upper}): {final_label_segments_arr.shape}")
            else:
                print(f"      Nessun segmento EEG valido generato per {subject_id}.")

    print("\n--- Generazione segmenti di training completata ---")