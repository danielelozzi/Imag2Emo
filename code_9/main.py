# main.py
import os
import json
import pandas as pd
import numpy as np
import optuna
import copy

# Import dai file del progetto
from data_loader import get_data_splits
from processa_deap_bdf import processa_e_salva_dati_deap_bdf
from processa_deap import processa_e_salva_dati_deap
from processa_graz import processa_e_salva_dati_graz
from prepare_training_segments import generate_training_segments
from eeg_classifier_training import train_and_evaluate_model
from reporting import plot_single_run_curves,plot_all_curves_in_one, plot_confusion_matrix, save_classification_report, plot_average_curves, save_summary_metrics, plot_class_distribution
from torch.utils.tensorboard import SummaryWriter

# La funzione run_hpo_and_get_best_trial rimane invariata
def run_hpo_and_get_best_trial(
    dataset_name, label_type, label_metric, model_type, apply_scaling,
    base_training_config, base_pipeline_config, sfreq_for_training, original_sfreq, dataset_info,
    results_base_path
):
    """
    Esegue l'ottimizzazione degli iperparametri (HPO) con Optuna per una specifica combinazione.
    Restituisce l'intero oggetto 'best_trial' trovato.
    """
    print(f"\n\n>>>>>> INIZIO HPO PER COMBINAZIONE SPECIFICA <<<<<<")
    print(f"Target: [{dataset_name}]-[{label_type}]-[{label_metric}]-[k_simple]-[{model_type}]-Scaling[{apply_scaling}]")

    def objective(trial):
        # Usiamo copie locali per non modificare le configurazioni globali durante l'HPO
        hpo_training_config = copy.deepcopy(base_training_config)
        hpo_pipeline_config = copy.deepcopy(base_pipeline_config)

        # --- Suggerimento dinamico degli iperparametri in base al model_type ---
        params_to_test = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        }

        if model_type == 'EEGNetv4':
            params_to_test['dropout'] = trial.suggest_float('dropout', 0.2, 0.7)
            params_to_test['F1'] = trial.suggest_int('F1', 8, 24, step=4)

        print(f"\n--- Inizio Trial HPO #{trial.number} ---")
        print(f"  Parametri in test per {model_type}: {params_to_test}")

        hpo_training_config['learning_rate'] = params_to_test['learning_rate']
        hpo_training_config['optimizer_params']['Adam']['weight_decay'] = params_to_test['weight_decay']

        if model_type == 'EEGNetv4':
            hpo_pipeline_config['model_params']['EEGNetv4']['dropout'] = params_to_test['dropout']
            hpo_pipeline_config['model_params']['EEGNetv4']['F1'] = params_to_test['F1']
            hpo_pipeline_config['model_params']['EEGNetv4']['F2'] = params_to_test['F1'] * hpo_pipeline_config['model_params']['EEGNetv4']['D']

        hpo_scenario = hpo_pipeline_config.get('hpo_training_scenario', 'k_simple')
        data_config = {
            'eeg_data_dir_raw': dataset_info['eeg_data_dir_raw'], 'labels_data_dir_base_raw': dataset_info['labels_base_dir_raw'],
            'slicing_config_dataset': dataset_info['slicing_config'], 'label_type': label_type, 'label_metric': label_metric,
            'training_scenario': hpo_scenario,
            **hpo_pipeline_config
        }
        
        hpo_training_config.update({
            'is_classification': data_config['binarized_threshold'] is not None,
            'sampling_rate': sfreq_for_training,
            'original_sampling_rate': original_sfreq,
            'apply_scaling': apply_scaling,
            'model_type': model_type,
            'model_params': hpo_pipeline_config.get('model_params', {}) # ADD THIS LINE
        })

        all_val_accuracies_for_trial = []
        data_generator_for_hpo_trial = get_data_splits(data_config, hpo_subset_ratio=hpo_pipeline_config['hpo_subset_ratio'])
        
        for fold_idx in range(hpo_pipeline_config['k_simple_repetitions']):
            try:
                X_train, y_train, X_val, y_val, X_test, y_test, info = next(data_generator_for_hpo_trial)
                hpo_training_config['fold_id'] = f"hpo_trial_{trial.number}_fold_{fold_idx}"
                data_splits_for_hpo_fold = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test}
                
                _, history = train_and_evaluate_model(hpo_training_config, data_splits_for_hpo_fold, skip_test_evaluation=True)
                best_val_accuracy_this_fold = max(history['val_accuracy']) if 'val_accuracy' in history and history['val_accuracy'] else 0.0
                all_val_accuracies_for_trial.append(best_val_accuracy_this_fold)
            except StopIteration:
                print(f"  Attenzione: Generatore di dati esaurito prima di {hpo_pipeline_config['k_simple_repetitions']} fold per il trial HPO {trial.number}. Interruzione.")
                break
            except Exception as e:
                print(f"  Errore durante il trial HPO {trial.number}, fold {fold_idx}: {e}. Saltando questo fold.")
                all_val_accuracies_for_trial.append(0.0)
        
        if not all_val_accuracies_for_trial:
            print(f"  Nessuna accuratezza valida raccolta per il trial HPO {trial.number}. Restituisco 0.0.")
            return 0.0

        average_val_accuracy = np.mean(all_val_accuracies_for_trial)
        print(f"--- Fine Trial HPO #{trial.number} | Media Accuratezza di Validazione su {len(all_val_accuracies_for_trial)} fold: {average_val_accuracy:.4f} ---")
        return average_val_accuracy

    n_hpo_trials = base_pipeline_config.get('hpo_n_trials', 5)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_hpo_trials)

    best_trial = study.best_trial
    print(f"\n>>>>>> HPO PER COMBINAZIONE COMPLETATO <<<<<<")
    print(f"Miglior Valore (Max Validation Accuracy): {best_trial.value:.4f}")
    print(f"Migliori Iperparametri: {best_trial.params}")

    hpo_studies_dir = os.path.join(results_base_path, "HPO_Studies")
    os.makedirs(hpo_studies_dir, exist_ok=True)
    study_df = study.trials_dataframe()
    scaling_str = "scaling_ON" if apply_scaling else "scaling_OFF"
    study_filename = f"hpo_study_{dataset_name}_{label_type}_{label_metric}_{model_type}_{scaling_str}.csv"
    study_filepath = os.path.join(hpo_studies_dir, study_filename)
    study_df.to_csv(study_filepath, index=False)
    print(f"Risultati dettagliati dello studio HPO salvati in: {study_filepath}")

    return best_trial


def main_orchestrator_function():
    """
    Funzione principale che orchestra l'intero pipeline di elaborazione dati e training.
    """
    print("--- Avvio del pipeline di elaborazione dati e training ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ##################################################################
    # ############## INTERRUTTORI DI CONTROLLO DEL PIPELINE ##############
    # ##################################################################
    # Imposta a False per saltare la generazione dei segmenti se già presenti su disco
    RUN_DISK_SEGMENTATION = False

    # Imposta a False per saltare l'HPO e caricare i parametri da un file JSON
    RUN_HPO = False
    # ##################################################################

    # --- PARAMETRI GLOBALI E PERCORSI ---
    DEAP_RAW_DATA_PATH = os.path.join(script_dir, "../../deap/deap_prep_set")
    DEAP_PARTICIPANT_RATING_PATH = os.path.join(script_dir, "./participant_ratings.xls")
    DEAP_VIDEO_LIST_PATH = os.path.join(script_dir, "./video_list.xls")
    
    DEAP_BDF_RAW_DATA_PATH = os.path.join(script_dir, "../../deap/PrePyt/data_preprocessed_python")
    DEAP_BDF_NPY_OUTPUT_BASE_PATH = os.path.join(script_dir, "./processed_data/DEAP_BDF_NPY")

    GRAZ_RAW_DATA_PATH = os.path.join(script_dir, "../graz_data/prep")
    GRAZ_LABELS_CSV_PATH = os.path.join(script_dir, "../label_private_filtered_p008_plus")
    GRAZ_SIMILAR_PAIRS_CSV = os.path.join(script_dir, "./similar_image_pairs.csv")

    DEAP_NPY_OUTPUT_BASE_PATH = os.path.join(script_dir, "./processed_data/DEAP_NPY")
    GRAZ_NPY_OUTPUT_BASE_PATH = os.path.join(script_dir, "./processed_data/GRAZ_NPY")
    
    NPY_TRAINING_OUTPUT_BASE_PATH_FOR_DISK_SAVE = os.path.join(script_dir, "./NPY_TRAINING_DISK_CACHE")

    TRAINING_RESULTS_OUTPUT_BASE_PATH = os.path.join(script_dir, "./TRAINING_RESULTS")
    os.makedirs(TRAINING_RESULTS_OUTPUT_BASE_PATH, exist_ok=True)
    
    # Percorso del file JSON da cui caricare i parametri se RUN_HPO = False
    OPTIMIZED_PARAMS_JSON_PATH = os.path.join(TRAINING_RESULTS_OUTPUT_BASE_PATH, "hpo_best_params.json")

    # --- Configurazione Slicing ---
    DEAP_TRAINING_PREP_CONFIG = {
        'start_time_idx': 5000, 'end_time_idx': 15000, 'channel_indices': list(range(32)),
        'n_channels': 32, 'sample_length_tp': 500,
    }
    GRAZ_TRAINING_PREP_CONFIG = {
        'start_time_idx': 500, 'end_time_idx': 2000, 'channel_indices': None,
        'n_channels': 32, 'sample_length_tp': 500,
    }
    DEAP_BDF_TRAINING_PREP_CONFIG = {
        'start_time_idx': 384, 'end_time_idx': 2560, 'channel_indices': list(range(32)),
        'n_channels': 32, 'sample_length_tp': 128,
    }



    # --- Informazioni Specifiche per Dataset ---
    dataset_specific_prep_info = {
        'DEAP_BDF': {
            'subject_ids': [f"s{i:02d}" for i in range(1, 23)], 'public_labels': ['valence_pubblica', 'arousal_pubblica'],
            'private_labels': ['valence', 'arousal'], 'sampling_rate': 128
        },
        'DEAP': {
            'subject_ids': [f"s{i:02d}" for i in range(1, 23)], 'public_labels': ['valence_pubblica', 'arousal_pubblica'],
            'private_labels': ['valence_privata', 'arousal_privata'], 'sampling_rate': 500
        },
        'GRAZ': {
            'subject_ids': [f"P{i:03d}" for i in range(8, 28)], 'public_labels': ['valence_pubblico', 'arousal_pubblico'],
            'private_labels': ['rate_valence_privata', 'rate_arousal_privato'], 'sampling_rate': 500
        }
    }

    # --- Configurazione Generale del Training ---
    TRAINING_CONFIG = {
        'epochs': 250, 'batch_size': 16, 'learning_rate': 0.0001,
        'early_stopping_patience': 20, 'factor_scheduler': 0.2, 'criterion_name': 'L1Loss',
        'optimizer_name': 'Adam',
        'optimizer_params': {
            'Adam': {'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-4},
            'SGD': {'momentum': 0.9, 'dampening': 0, 'weight_decay': 1e-4, 'nesterov': False}
        }
    }
    
    # --- Configurazione del Pipeline ---
    PIPELINE_CONFIG = {
        'kfold_splits': 5, 'loso_val_subjects': 2, 'binarized_threshold': 5.0,
        'balancing_strategy': 'custom_undersampling', 'loso_test_limit': None, 
        'k_simple_repetitions': 5, 'global_shuffle_seed': 42,
        'model_types': ['EEGNetv4'], 'apply_resample': True, 'new_sampling_rate': 128,
        'apply_butter_filter': True, 'butter_l_freq': 4.0, 'butter_h_freq': 64.0, 'butter_order': 4,
        'hpo_subset_ratio': 0.3, 'hpo_n_trials': 5, 'hpo_training_scenario': 'k_simple',
        'model_params': {
            'ContraNet': {
                'kernLength': 250, 'poolLength': 8, 'numFilters': 16, 'projection_dim': 32,
                'transformer_layers': 1, 'num_heads': 8, 'transformer_units': [64, 32], 'mlp_head_units': [112]
            },
            'EEGNetv4': {
                'F1': 16, 'D': 2, 'F2': 32, 'kernel_length': 128, 'third_kernel_size': (8, 4), 'dropout': 0.4
            },
            'Conformer': {
                'emb_size': 40, 'depth': 6, 'n_heads': 10, 'patch_size': 25, 'pool_size': 75,
                'pool_stride': 15, 'num_shallow_filters': 40, 'drop_p': 0.5, 'forward_expansion': 4, 'forward_drop_p': 0.5
            },
            'ERTNet': {'dropoutRate': 0.5, 'kernLength': 64, 'F1': 8, 'heads': 8, 'D': 2, 'F2': 16},
            'EEGDeformer': {'temporal_kernel': 11, 'num_kernel': 64, 'depth': 4, 'heads': 16, 'mlp_dim': 16, 'dim_head': 16, 'dropout': 0.0},
            'EEGNet_custom': {'dropoutP': 0.25, 'F1': 8, 'D': 2, 'C1': 64},
            'EEGViT': {'num_patches': 24, 'dim': 32, 'depth': 4, 'heads': 16, 'mlp_dim': 64, 'pool': 'cls', 'dim_head': 64, 'dropout': 0.1, 'emb_dropout': 0.1} 
        }
    }
    
    PIPELINE_CONFIG['model_types'] = ['EEGNetv4','ContraNet','EEGViT','EEGDeformer','Conformer']

    if PIPELINE_CONFIG.get('apply_resample', False):
        new_sfreq = PIPELINE_CONFIG['new_sampling_rate']
        PIPELINE_CONFIG['model_params']['ContraNet']['kernLength'] = int(new_sfreq)
        PIPELINE_CONFIG['model_params']['EEGNetv4']['kernel_length'] = int(new_sfreq // 2)
        print(f"\n--- Parametro 'kernel_length' di EEGNetv4 aggiornato dinamicamente a: {PIPELINE_CONFIG['model_params']['EEGNetv4']['kernel_length']} ---")
        target_patch_length_for_eegvit = 16
        calculated_num_patches_eegvit = new_sfreq // target_patch_length_for_eegvit
        if calculated_num_patches_eegvit == 0 or new_sfreq % calculated_num_patches_eegvit != 0:
            calculated_num_patches_eegvit = 8 if new_sfreq == 128 else 1
            print(f"  ATTENZIONE: num_patches per EEGViT non è un divisore ideale per {new_sfreq}. Usando {calculated_num_patches_eegvit}.")
        PIPELINE_CONFIG['model_params']['EEGViT']['num_patches'] = calculated_num_patches_eegvit
        print(f"\n--- Parametro 'num_patches' di EEGViT aggiornato dinamicamente a: {PIPELINE_CONFIG['model_params']['EEGViT']['num_patches']} ---")
        print(f"\n--- Parametro 'kernLength' di ContraNet aggiornato dinamicamente a: {PIPELINE_CONFIG['model_params']['ContraNet']['kernLength']} ---")

    # --- Configurazione Dati per il Training ---
    datasets_to_train = {
        'GRAZ': {
            'eeg_data_dir_raw': os.path.join(NPY_TRAINING_OUTPUT_BASE_PATH_FOR_DISK_SAVE, "GRAZ_NPY", "EEG"),
            'labels_base_dir_raw': os.path.join(NPY_TRAINING_OUTPUT_BASE_PATH_FOR_DISK_SAVE, "GRAZ_NPY", "LABEL"),
            'public_labels': ['valence_pubblico', 'arousal_pubblico'], 'private_labels': ['rate_valence_privata', 'rate_arousal_privato'],
            'sampling_rate': 500, 'slicing_config': GRAZ_TRAINING_PREP_CONFIG, 'apply_scaling_options': [True]
        },
        'DEAP_BDF': {
            'eeg_data_dir_raw': os.path.join(NPY_TRAINING_OUTPUT_BASE_PATH_FOR_DISK_SAVE, "DEAP_BDF_NPY", "EEG"),
            'labels_base_dir_raw': os.path.join(NPY_TRAINING_OUTPUT_BASE_PATH_FOR_DISK_SAVE, "DEAP_BDF_NPY", "LABEL"),
            'public_labels': ['valence_pubblica', 'arousal_pubblica'], 'private_labels': ['valence', 'arousal'],
            'sampling_rate': 128, 'slicing_config': DEAP_BDF_TRAINING_PREP_CONFIG, 'apply_scaling_options': [False]
        },
        'DEAP': {
            'eeg_data_dir_raw': os.path.join(NPY_TRAINING_OUTPUT_BASE_PATH_FOR_DISK_SAVE, "DEAP_NPY", "EEG"),
            'labels_base_dir_raw': os.path.join(NPY_TRAINING_OUTPUT_BASE_PATH_FOR_DISK_SAVE, "DEAP_NPY", "LABEL"),
            'public_labels': ['valence_pubblica', 'arousal_pubblica'], 'private_labels': ['valence_privata', 'arousal_privata'],
            'sampling_rate': 500, 'slicing_config': DEAP_TRAINING_PREP_CONFIG, 'apply_scaling_options': [True]
        }
    }


    # --- Salvataggio Configurazione Iniziale ---
    full_config_summary = {
        'PIPELINE_EXECUTION_CONTROL': {'RUN_DISK_SEGMENTATION': RUN_DISK_SEGMENTATION, 'RUN_HPO': RUN_HPO},
        'DEAP_TRAINING_PREP_CONFIG': DEAP_TRAINING_PREP_CONFIG, 'GRAZ_TRAINING_PREP_CONFIG': GRAZ_TRAINING_PREP_CONFIG,
        'dataset_specific_prep_info': dataset_specific_prep_info, 'TRAINING_CONFIG': TRAINING_CONFIG,
        'PIPELINE_CONFIG': PIPELINE_CONFIG, 'datasets_to_train': datasets_to_train
    }
    config_output_path = os.path.join(TRAINING_RESULTS_OUTPUT_BASE_PATH, "experiment_config_summary.json")
    with open(config_output_path, 'w') as f: json.dump(full_config_summary, f, indent=4)
    print(f"\n--- Configurazione iniziale dell'esperimento salvata in: {config_output_path} ---")

    # --- STEP 1-4: Preprocessing e Segmentazione ---
    if RUN_DISK_SEGMENTATION:
        print("\n##### Avvio Fase di Preprocessing e Segmentazione su Disco #####")
        try:
            processa_e_salva_dati_deap(DEAP_RAW_DATA_PATH, DEAP_NPY_OUTPUT_BASE_PATH, DEAP_PARTICIPANT_RATING_PATH, DEAP_VIDEO_LIST_PATH)
        except Exception as e: print(f"ERRORE durante l'elaborazione dati DEAP (originale): {e}")
        try:
            processa_e_salva_dati_deap_bdf(DEAP_BDF_RAW_DATA_PATH, DEAP_BDF_NPY_OUTPUT_BASE_PATH, DEAP_PARTICIPANT_RATING_PATH, DEAP_VIDEO_LIST_PATH)
        except Exception as e: print(f"ERRORE durante l'elaborazione dati DEAP_BDF: {e}")
        try:
            processa_e_salva_dati_graz(GRAZ_RAW_DATA_PATH, GRAZ_LABELS_CSV_PATH, GRAZ_SIMILAR_PAIRS_CSV, GRAZ_NPY_OUTPUT_BASE_PATH)
        except Exception as e: print(f"ERRORE durante l'elaborazione dati GRAZ: {e}")
            
        os.makedirs(NPY_TRAINING_OUTPUT_BASE_PATH_FOR_DISK_SAVE, exist_ok=True)
        try:
            generate_training_segments(
                processed_data_base_paths={'DEAP': DEAP_NPY_OUTPUT_BASE_PATH, 'DEAP_BDF': DEAP_BDF_NPY_OUTPUT_BASE_PATH, 'GRAZ': GRAZ_NPY_OUTPUT_BASE_PATH},
                training_output_base_path=NPY_TRAINING_OUTPUT_BASE_PATH_FOR_DISK_SAVE,
                slicing_configs={'DEAP': DEAP_TRAINING_PREP_CONFIG, 'DEAP_BDF': DEAP_BDF_TRAINING_PREP_CONFIG, 'GRAZ': GRAZ_TRAINING_PREP_CONFIG},
                dataset_specific_info=dataset_specific_prep_info, pipeline_config=PIPELINE_CONFIG
            )
        except Exception as e: print(f"ERRORE durante la generazione dei segmenti di training: {e}")
    else:
        print("\n##### Fase di Preprocessing e Segmentazione su Disco SALTATA come da configurazione #####")

    
    # --- FASE 1: HPO O CARICAMENTO PARAMETRI ---
    sfreq_for_training_global = PIPELINE_CONFIG.get('new_sampling_rate') if PIPELINE_CONFIG.get('apply_resample') else None
    optimized_params_store_per_dataset = {}

    if RUN_HPO:
        print("\n\n##### Avvio Fase 1: Ottimizzazione Iperparametri (HPO) per Dataset #####")
        for dataset_name, dataset_info in datasets_to_train.items():
            original_sfreq = dataset_info['sampling_rate']
            sfreq_for_training = sfreq_for_training_global if sfreq_for_training_global is not None else original_sfreq
            
            if sfreq_for_training != original_sfreq: print(f"[{dataset_name}] Applicherà ricampionamento: {original_sfreq} Hz -> {sfreq_for_training} Hz.")
            dataset_info['slicing_config']['sampling_rate'] = original_sfreq
            all_best_trials_for_dataset = []

            for label_type in ['PRIVATE', 'PUBLIC']:
                if not dataset_info[f'{label_type.lower()}_labels']: continue
                for label_metric in dataset_info[f'{label_type.lower()}_labels']:
                    model_types_to_run = PIPELINE_CONFIG['model_types'] if isinstance(PIPELINE_CONFIG['model_types'], list) else [PIPELINE_CONFIG['model_types']]
                    for model_type in model_types_to_run:
                        for apply_scaling in dataset_info['apply_scaling_options']:
                            best_trial_for_combo = run_hpo_and_get_best_trial(
                                dataset_name, label_type, label_metric, model_type, apply_scaling,
                                TRAINING_CONFIG, PIPELINE_CONFIG, sfreq_for_training, original_sfreq, dataset_info,
                                TRAINING_RESULTS_OUTPUT_BASE_PATH
                            )
                            all_best_trials_for_dataset.append(best_trial_for_combo)

            if all_best_trials_for_dataset:
                overall_best_trial = max(all_best_trials_for_dataset, key=lambda trial: trial.value)
                optimized_params_store_per_dataset[dataset_name] = overall_best_trial.params
                print(f"\n\n===== MIGLIORI IPERPARAMETRI COMPLESSIVI PER IL DATASET '{dataset_name}' =====")
                print(f"  Trovati dal trial con la più alta validation accuracy: {overall_best_trial.value:.4f}")
                print(f"  Parametri: {overall_best_trial.params}")
                print("=" * (60 + len(dataset_name)))
            else:
                print(f"ATTENZIONE: Nessun HPO completato con successo per il dataset {dataset_name}. Verranno usati i parametri di default.")
                optimized_params_store_per_dataset[dataset_name] = {}
        
        # Salva i parametri ottimizzati in un file JSON pulito per un facile riutilizzo
        print(f"\n[RIEPILOGO HPO] Salvataggio dei migliori iperparametri trovati in: {OPTIMIZED_PARAMS_JSON_PATH}")
        with open(OPTIMIZED_PARAMS_JSON_PATH, 'w') as f:
            json.dump(optimized_params_store_per_dataset, f, indent=4)
    
    else: # Se RUN_HPO è False, carica i parametri
        print("\n\n##### Fase 1: HPO Saltata. Caricamento parametri ottimizzati... #####")
        try:
            with open(OPTIMIZED_PARAMS_JSON_PATH, 'r') as f:
                optimized_params_store_per_dataset = json.load(f)
            print(f"  Parametri caricati con successo da: {OPTIMIZED_PARAMS_JSON_PATH}")
            print("  Parametri caricati:", json.dumps(optimized_params_store_per_dataset, indent=2))
        except FileNotFoundError:
            print(f"  ATTENZIONE: File dei parametri '{OPTIMIZED_PARAMS_JSON_PATH}' non trovato.")
            print("  Il training procederà con i parametri di default definiti nella configurazione.")
            optimized_params_store_per_dataset = {}
        except json.JSONDecodeError:
            print(f"  ERRORE: Impossibile decodificare il file JSON '{OPTIMIZED_PARAMS_JSON_PATH}'.")
            print("  Il training procederà con i parametri di default.")
            optimized_params_store_per_dataset = {}


    # --- FASE 2: ESECUZIONE DEL TRAINING COMPLETO CON PARAMETRI OTTIMIZZATI ---
    print("\n\n##### Avvio Fase 2: Training Completo con Parametri Ottimizzati #####")
    for dataset_name, dataset_info in datasets_to_train.items():
        original_sfreq = dataset_info['sampling_rate']
        sfreq_for_training = sfreq_for_training_global if sfreq_for_training_global is not None else original_sfreq
        dataset_info['slicing_config']['sampling_rate'] = original_sfreq
        best_params = optimized_params_store_per_dataset.get(dataset_name, {})
        
        for label_type in ['PRIVATE', 'PUBLIC']:
            for label_metric in dataset_info[f'{label_type.lower()}_labels']:
                for scenario in ['k_simple', 'kfold', 'loso']:
                    model_types_to_run = PIPELINE_CONFIG['model_types'] if isinstance(PIPELINE_CONFIG['model_types'], list) else [PIPELINE_CONFIG['model_types']]
                    for model_type in model_types_to_run:
                        for apply_scaling in dataset_info['apply_scaling_options']:
                            current_training_config = copy.deepcopy(TRAINING_CONFIG)
                            current_pipeline_config = copy.deepcopy(PIPELINE_CONFIG)
                            current_pipeline_config['hpo_subset_ratio'] = 1.0

                            combination_key = (dataset_name, label_type, label_metric, scenario, model_type, apply_scaling)
                            if best_params:
                                print(f"\n[APPLICAZIONE PARAMETRI OTTIMIZZATI] per {combination_key}:")
                                if 'learning_rate' in best_params:
                                    current_training_config['learning_rate'] = best_params['learning_rate']
                                    print(f"  -> LR: {best_params['learning_rate']}")
                                if 'weight_decay' in best_params:
                                    current_training_config['optimizer_params']['Adam']['weight_decay'] = best_params['weight_decay']
                                    print(f"  -> Weight Decay: {best_params['weight_decay']}")
                                
                                model_to_tune = 'EEGNetv4'
                                if 'dropout' in best_params:
                                    current_pipeline_config['model_params'][model_to_tune]['dropout'] = best_params['dropout']
                                    print(f"  -> Dropout ({model_to_tune}): {best_params['dropout']}")
                                if 'F1' in best_params:
                                    current_pipeline_config['model_params'][model_to_tune]['F1'] = best_params['F1']
                                    current_pipeline_config['model_params'][model_to_tune]['F2'] = current_pipeline_config['model_params'][model_to_tune]['F1'] * current_pipeline_config['model_params'][model_to_tune]['D']
                                    print(f"  -> F1/F2 ({model_to_tune}) aggiornati.")
                            else:
                                print(f"\n[INFO] Nessun parametro ottimizzato trovato per {combination_key}. Verranno usati i parametri di default.")

                            scaling_str = "scaling_ON" if apply_scaling else "scaling_OFF"
                            run_name = f"[{dataset_name}]-[{label_type}]-[{label_metric}]-[{scenario}]-[{model_type}]-[{scaling_str}]"
                            print(f"\n>>>>>> INIZIO SCENARIO: {run_name} <<<<<<")

                            output_dir = os.path.join(TRAINING_RESULTS_OUTPUT_BASE_PATH, dataset_name, label_type, label_metric, scenario, model_type, scaling_str)
                            os.makedirs(output_dir, exist_ok=True)
                            
                            data_config = {
                                'eeg_data_dir_raw': dataset_info['eeg_data_dir_raw'], 'labels_data_dir_base_raw': dataset_info['labels_base_dir_raw'],
                                'slicing_config_dataset': dataset_info['slicing_config'], 'label_type': label_type, 'label_metric': label_metric,
                                'training_scenario': scenario, **current_pipeline_config
                            }
                            
                            current_training_config.update({
                                'is_classification': data_config['binarized_threshold'] is not None,
                                'sampling_rate': sfreq_for_training,
                                'original_sampling_rate': original_sfreq,
                                'apply_scaling': apply_scaling,
                                'model_type': model_type,
                                'model_params': current_pipeline_config.get('model_params', {}) # ADD THIS LINE
                            })
                            
                            try:
                                fold_results, fold_histories, fold_reports = [], [], []
                                data_generator = get_data_splits(data_config)
                                tensorboard_run_dir = os.path.join(output_dir, "tensorboard_logs")
                                print(f"  La cartella base per i log di TensorBoard di questo run è: {tensorboard_run_dir}")

                                for X_train, y_train, X_val, y_val, X_test, y_test, info in data_generator:
                                    fold_id = info.get('fold')
                                    print(f"Fold {fold_id} Dati Caricati: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

                                    writer = SummaryWriter(log_dir=os.path.join(tensorboard_run_dir, f"fold_{fold_id}"))

                                    if scenario == 'loso':
                                        loso_info_filepath = os.path.join(output_dir, f"loso_fold_{fold_id}_subject_split.json")
                                        with open(loso_info_filepath, 'w') as f_json:
                                            json.dump({'test_subject': info.get('test_subject'), 'validation_subjects': info.get('val_subjects'), 'training_subjects': info.get('train_subjects')}, f_json, indent=4)

                                    if current_training_config['is_classification']:
                                        plot_class_distribution(y_train, f"{run_name} Train Fold {fold_id}", os.path.join(output_dir, f"dist_train_fold_{fold_id}.png"))
                                        plot_class_distribution(y_val, f"{run_name} Val Fold {fold_id}", os.path.join(output_dir, f"dist_val_fold_{fold_id}.png"))
                                        plot_class_distribution(y_test, f"{run_name} Test Fold {fold_id}", os.path.join(output_dir, f"dist_test_fold_{fold_id}.png"))
                                    
                                    current_training_config['fold_id'] = fold_id
                                    data_splits = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test}
                                    
                                    metrics, history = train_and_evaluate_model(current_training_config, data_splits, writer=writer)
                                    writer.close()
                                    
                                    fold_histories.append(history)
                                    plot_single_run_curves(history, f"{run_name} Fold {fold_id}", os.path.join(output_dir, f"loss_curve_fold_{fold_id}.png"))
                                    
                                    history_path = os.path.join(output_dir, f"history_fold_{fold_id}.json")
                                    with open(history_path, 'w') as f: json.dump(history, f, indent=4)

                                    if metrics:
                                        metrics['fold'] = fold_id
                                        report_dict = metrics.pop('report', None)
                                        conf_matrix_data = metrics.pop('conf_matrix_data', None)
                                        fold_results.append(metrics)
                                        
                                        if report_dict:
                                            fold_reports.append(report_dict)
                                            save_classification_report(report_dict, os.path.join(output_dir, f"report_fold_{fold_id}.json"))
                                        
                                        if conf_matrix_data and len(conf_matrix_data[0]) > 0 and len(conf_matrix_data[1]) > 0:
                                            y_true_cm, y_pred_cm = conf_matrix_data
                                            all_present_labels = sorted(list(set(y_true_cm) | set(y_pred_cm)))
                                            plot_confusion_matrix(y_true_cm, y_pred_cm, all_present_labels, f"{run_name} Fold {fold_id}", os.path.join(output_dir, f"confusion_matrix_fold_{fold_id}.png"))

                                if scenario in ['kfold', 'loso', 'k_simple'] and fold_results:
                                    results_df = pd.DataFrame(fold_results)
                                    save_summary_metrics(results_df, os.path.join(output_dir, "summary_metrics.csv"), reports_list=fold_reports)
                                    
                                    plot_average_curves(fold_histories, 'train_loss', run_name, os.path.join(output_dir, "avg_train_loss.png"))
                                    plot_average_curves(fold_histories, 'val_loss', run_name, os.path.join(output_dir, "avg_val_loss.png"))
                                    if current_training_config['is_classification']:
                                        plot_average_curves(fold_histories, 'val_accuracy', run_name, os.path.join(output_dir, "avg_val_accuracy.png"))
                                    plot_all_curves_in_one(fold_histories, run_name, os.path.join(output_dir, "avg_curves.png"))
                            except FileNotFoundError as e:
                                print(f"ERRORE File Non Trovato per {run_name}: {e}. Saltando.")
                            except Exception as e:
                                print(f"ERRORE Generico durante il training/valutazione per {run_name}: {e}. Saltando.")

    print("\n--- Pipeline di training e reporting completato. ---")

if __name__ == '__main__':
    main_orchestrator_function()