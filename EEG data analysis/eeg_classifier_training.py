# eeg_classifier_training.py
import torch
import warnings
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from braindecode.models import EEGNetv4
from sklearn.metrics import (classification_report, f1_score, precision_score, 
                             recall_score, accuracy_score, r2_score, 
                             mean_squared_error, mean_absolute_error)
import copy
import eegmodels
import mne

def compute_max(shape,batch_size):
    r = shape-shape%batch_size
    return r

def train_and_evaluate_model(config: dict, data_splits: dict, writer=None, skip_test_evaluation: bool = False):
    """
    Esegue il training, la validazione e il test di un modello.

    Args:
        config (dict): Dizionario di configurazione.
        data_splits (dict): Dati di train, validation, e test.
        writer (torch.utils.tensorboard.SummaryWriter, optional): Writer per TensorBoard. Default a None.
        skip_test_evaluation (bool): Se True, salta la valutazione finale sul test set.
    """
    # --- 1. Preparazione Dati e Parametri ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_classification = config['is_classification']
    apply_scaling = config.get('apply_scaling', True)
    try:
        model_type = config['model_type']
    except KeyError:
        raise KeyError("La chiave 'model_type' è obbligatoria nella configurazione.")
    fold_id_desc = config.get('fold_id')

    X_train, y_train_orig = data_splits['X_train'], data_splits['y_train']
    X_val, y_val_orig = data_splits['X_val'], data_splits['y_val']
    X_test, y_test = data_splits['X_test'], data_splits['y_test']

    # --- 4. Scaling Dati ---
    if apply_scaling:
        if X_train.shape[0] > 0:
            scaler = mne.decoding.Scaler(scalings='mean')
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            if X_val.shape[0] > 0:
                X_val = scaler.transform(X_val)
            if X_test.shape[0] > 0:
                X_test = scaler.transform(X_test)
        else:
            print(f"  ATTENZIONE: X_train è vuoto per fold {fold_id_desc}. Scaling saltato.")
    
    # --- Calcolo robusto del numero di classi ---
    all_labels_in_fold = np.concatenate([arr for arr in [y_train_orig, y_val_orig, y_test] if arr.size > 0])

    # Se is_classification è True, calcola n_classes dai dati binarizzati.
    # Altrimenti, per regressione, n_classes potrebbe non avere senso o dovrebbe essere 1.
    if is_classification:
        if all_labels_in_fold.size > 0:
            n_classes = len(np.unique(all_labels_in_fold))
            print(f"  Numero di classi trovate nei dati: {n_classes} (Fold {fold_id_desc})")
            if n_classes < 2:
                # Questo può accadere in subset molto piccoli o con dati sbilanciati
                print(f"  ATTENZIONE (Fold {fold_id_desc}): Trovata meno di 2 classi nei dati binarizzati. Imposto n_classes=2.")
                n_classes = 2 # Minimo 2 classi per classificazione
        else:
            print(f"  ATTENZIONE (Fold {fold_id_desc}): Dati di etichette binarizzate vuoti. Imposto n_classes=2.")
            n_classes = 2
    else: # Regression
        n_classes = 1 # Output neuron for regression

    # Assicurati che y_train e y_val siano one-hot encoded SOLO per classificazione
    if is_classification:
        y_train = torch.nn.functional.one_hot(torch.LongTensor(y_train_orig), n_classes).float()
        y_val = torch.nn.functional.one_hot(torch.LongTensor(y_val_orig), n_classes).float()
    else:
        # Per regressione, le label dovrebbero essere float e non one-hot
        y_train = torch.Tensor(y_train_orig).float()
        y_val = torch.Tensor(y_val_orig).float()
        # Se le label hanno una dimensione in più (es. (N, 1)), appiattiscile a (N,)
        if y_train.ndim > 1 and y_train.shape[1] == 1:
            y_train = y_train.squeeze(1)
        if y_val.ndim > 1 and y_val.shape[1] == 1:
            y_val = y_val.squeeze(1)


    # --- 5. Definizione del Modello ---
    n_channels = X_train.shape[1]
    input_window_samples = X_train.shape[2]
    sfreq = config['sampling_rate']
    
    if model_type == 'EEGNetv4':
        params = config.get('model_params', {}).get('EEGNetv4', {})
        model = EEGNetv4(n_chans=n_channels, n_outputs=n_classes, n_times=input_window_samples, sfreq=sfreq, **params).to(device)
    elif model_type == 'ContraNet':
        params = config['model_params']['ContraNet']
        model = eegmodels.contranet(nb_classes=n_classes, Chans=n_channels, Samples=input_window_samples, **params).to(device)
    elif model_type == 'Conformer':
        params = config.get('model_params', {}).get('Conformer', {})
        patch_size, pool_size, pool_stride, emb_size = params.get('patch_size', 25), params.get('pool_size', 75), params.get('pool_stride', 15), params.get('emb_size', 40)
        length_after_conv = input_window_samples - patch_size + 1
        w = (length_after_conv - pool_size) // pool_stride + 1
        if w <= 0: raise ValueError(f"Conformer: Numero di patch calcolato (w={w}) non valido.")
        calculated_n_hidden = w * emb_size
        model = eegmodels.conformer(n_chan=n_channels, n_classes=n_classes, n_times=input_window_samples, custom_n_hidden=calculated_n_hidden, **params).to(device)
    elif model_type == 'ERTNet':
        params = config.get('model_params', {}).get('ERTNet', {})
        model = eegmodels.ertnet(nb_classes=n_classes, Chans=n_channels, Samples=input_window_samples, **params).to(device)
    elif model_type == 'EEGDeformer':
        params = config.get('model_params', {}).get('EEGDeformer', {})
        model = eegmodels.eegdeformer(num_chan=n_channels, num_time=input_window_samples, num_classes=n_classes, **params).to(device)
    elif model_type == 'EEGNet_custom':
        params = config.get('model_params', {}).get('EEGNet_custom', {})
        model = eegmodels.eegnet(nChan=n_channels, nTime=input_window_samples, nClass=n_classes, **params).to(device)
    elif model_type == 'EEGViT':
        params = config.get('model_params', {}).get('EEGViT', {})
        model = eegmodels.eegvit(num_chan=n_channels, num_time=input_window_samples, num_classes=n_classes, **params).to(device)
    else:
        raise ValueError(f"Tipo di modello '{model_type}' non supportato.")

    for param in model.parameters():
        param.requires_grad = True

    # --- 6. Definizione di Loss, Optimizer e Scheduler ---
    criterion_name = config.get('criterion_name', 'CrossEntropyLoss')
    optimizer_name = config.get('optimizer_name', 'Adam')
    
    # Se è classificazione e il criterio non è CrossEntropy, avvisa.
    if is_classification and criterion_name != 'CrossEntropyLoss':
        warnings.warn(f"Criterio '{criterion_name}' usato per classificazione. Si raccomanda 'CrossEntropyLoss'.")
    elif not is_classification and criterion_name == 'CrossEntropyLoss':
        warnings.warn(f"Criterio '{criterion_name}' usato per regressione. Si raccomanda 'L1Loss' o 'MSELoss'.")

    criterion = getattr(nn, criterion_name)()
    specific_optimizer_params = config.get('optimizer_params', {}).get(optimizer_name, {})
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config['learning_rate'], **specific_optimizer_params)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config['early_stopping_patience'], factor=config['factor_scheduler'])

    max_batch_X_train = compute_max(X_train.shape[0], config['batch_size'])
    max_batch_X_val = compute_max(X_val.shape[0], config['batch_size'])

    # --- 7. Loop di Training e Validazione ---
    best_val_loss = float('inf')
    epoch_since_best = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []} # 'val_accuracy' rinominato per generalità
    
    print(f"Inizio training per fold {fold_id_desc} su {device} | Classificazione: {is_classification} | Classi: {n_classes}")

    batch_size = config['batch_size']
    num_epochs = config['epochs']
    
    for epoch in range(num_epochs):
        model.train()
        total_training_loss = 0
        train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), y_train), batch_size=batch_size, shuffle=True)
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Per regressione, assicurati che i target abbiano la stessa dimensione delle predizioni (se necessario, squeeze)
            if not is_classification and targets.ndim == 1:
                targets = targets.unsqueeze(1) # Aggiunge dimensione per batch_size, 1

            outputs_train = model(inputs)
            
            # Se la loss function è MSELoss o L1Loss e l'output del modello ha una dimensione in più (es. (batch_size, 1)),
            # assicurati che anche i target abbiano la stessa dimensione.
            # Questo è più robusto se il modello di regressione emette (batch_size, 1)
            if not is_classification and outputs_train.shape != targets.shape:
                 targets = targets.view_as(outputs_train)

            loss = criterion(outputs_train, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item()
        
        avg_training_loss = total_training_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        history['train_loss'].append(avg_training_loss)

        model.eval()
        total_val_loss = 0.0
        all_val_preds = []
        all_val_targets_orig = []
        val_loader = DataLoader(TensorDataset(torch.Tensor(X_val), y_val), batch_size=batch_size)
        
        with torch.no_grad():
            for inputs, targets_val in val_loader:
                inputs, targets_val = inputs.to(device), targets_val.to(device)
                
                if not is_classification and targets_val.ndim == 1:
                    targets_val = targets_val.unsqueeze(1)

                outputs_val = model(inputs)

                if not is_classification and outputs_val.shape != targets_val.shape:
                    targets_val = targets_val.view_as(outputs_val)

                loss = criterion(outputs_val, targets_val)
                total_val_loss += loss.item()
                
                if is_classification:
                    _, predicted_indices = torch.max(outputs_val.data, 1)
                    all_val_preds.append(predicted_indices)
                    _, targets_indices_batch = torch.max(targets_val, 1)
                    all_val_targets_orig.append(targets_indices_batch)
                else: # Regression
                    all_val_preds.append(outputs_val.cpu().numpy().flatten())
                    all_val_targets_orig.append(targets_val.cpu().numpy().flatten())


            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            history['val_loss'].append(avg_val_loss)
            
            # Calcolo delle metriche di validazione
            if is_classification:
                if all_val_preds:
                    all_val_preds = torch.cat(all_val_preds).cpu().numpy()
                    all_val_targets_orig = torch.cat(all_val_targets_orig).cpu().numpy()
                    accuracy = accuracy_score(all_val_targets_orig, all_val_preds)
                else:
                    accuracy = 0.0
                history['val_metrics'].append(accuracy) # Aggiungi come accuracy per classificazione
            else: # Regression
                if all_val_preds:
                    all_val_preds = np.concatenate(all_val_preds)
                    all_val_targets_orig = np.concatenate(all_val_targets_orig)
                    r2 = r2_score(all_val_targets_orig, all_val_preds)
                    history['val_metrics'].append(r2) # Aggiungi come R2 per regressione
                else:
                    history['val_metrics'].append(0.0) # O np.nan, a seconda della preferenza

            # Early stopping basato sulla val_loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epoch_since_best = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                epoch_since_best += 1
                if epoch_since_best >= config['early_stopping_patience']:
                    print(f'Early stopping at epoch {epoch+1}.')
                    break
            scheduler.step(avg_val_loss)
        
        current_metric_value = history['val_metrics'][-1] if history['val_metrics'] else 0.0
        metric_name = "Val Acc" if is_classification else "Val R2"
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_training_loss:.6f}, Val Loss: {avg_val_loss:.6f}, {metric_name}: {current_metric_value:.4f}")

        if writer:
            writer.add_scalar('Loss/Train', avg_training_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            if is_classification:
                writer.add_scalar('Accuracy/Validation', current_metric_value, epoch)
            else:
                writer.add_scalar('R2/Validation', current_metric_value, epoch) # Aggiungi R2 per regressione
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

    if not skip_test_evaluation:
        # --- 8. Calcolo delle Metriche Finali sul Test Set ---
        if X_test is None or y_test is None or X_test.shape[0] == 0:
            print("  ATTENZIONE: Test set vuoto. Nessuna metrica finale calcolata.")
            return {}, history

        if best_model_weights:
            model.load_state_dict(best_model_weights)

        model.eval()
        test_y_pred_list = []
        # Per il test set, usa un batch_size più grande se la RAM lo permette, o rimane 1 per coerenza
        test_loader = DataLoader(TensorDataset(torch.Tensor(X_test)), batch_size=config['batch_size']) # Usa batch_size del config
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs_test = model(inputs)
                if is_classification:
                    test_y_pred_list.append(torch.argmax(outputs_test, dim=1).cpu().numpy().flatten())
                else: # Regression
                    test_y_pred_list.append(outputs_test.cpu().numpy().flatten())
                
        y_pred = np.concatenate(test_y_pred_list) if test_y_pred_list else np.array([])
        y_test_final_eval = y_test[:len(y_pred)] # Assicurati che y_test abbia la stessa lunghezza delle predizioni

        if y_test_final_eval.size == 0:
            print("  ATTENZIONE: Nessuna predizione generata per il test set. Metriche finali non calcolate.")
            return {}, history

        final_metrics = {}
        if is_classification:
            final_metrics['accuracy'] = accuracy_score(y_test_final_eval, y_pred)
            final_metrics['f1_score'] = f1_score(y_test_final_eval, y_pred, average='weighted', zero_division=0)
            final_metrics['precision'] = precision_score(y_test_final_eval, y_pred, average='weighted', zero_division=0)
            final_metrics['recall'] = recall_score(y_test_final_eval, y_pred, average='weighted', zero_division=0)
            final_metrics['report'] = classification_report(y_test_final_eval, y_pred, zero_division=0, output_dict=True)
            final_metrics['conf_matrix_data'] = (y_test_final_eval, y_pred)
        else: # Regression metrics
            final_metrics['r2_score'] = r2_score(y_test_final_eval, y_pred)
            final_metrics['mse'] = mean_squared_error(y_test_final_eval, y_pred)
            final_metrics['mae'] = mean_absolute_error(y_test_final_eval, y_pred)


        print(final_metrics)

        if writer and final_metrics:
            hparams = {
                'lr': config['learning_rate'],
                'batch_size': config['batch_size'],
                'model_type': config['model_type'],
                'scaling': str(apply_scaling)
            }
            if is_classification:
                final_metrics_for_hparams = {
                    'hparam/accuracy': final_metrics.get('accuracy', 0),
                    'hparam/f1_score': final_metrics.get('f1_score', 0)
                }
            else: # Regression
                final_metrics_for_hparams = {
                    'hparam/r2_score': final_metrics.get('r2_score', 0),
                    'hparam/mse': final_metrics.get('mse', 0)
                }
            writer.add_hparams(hparams, final_metrics_for_hparams)

        return final_metrics, history

    return {}, history # Se skip_test_evaluation è True, restituisce metriche vuote