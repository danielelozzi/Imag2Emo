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

    if all_labels_in_fold.size > 0:
        n_classes = len(np.unique(all_labels_in_fold))
        if is_classification and n_classes < 2:
            print(f"  ATTENZIONE (Fold {fold_id_desc}): Trovata solo 1 classe nei dati. Imposto n_classes=2 per coerenza.")
            n_classes = 2
    else:
        n_classes = 2

    y_train = torch.nn.functional.one_hot(torch.LongTensor(y_train_orig), n_classes).float()
    y_val = torch.nn.functional.one_hot(torch.LongTensor(y_val_orig), n_classes).float()
    
    # --- 5. Definizione del Modello ---
    n_channels = X_train.shape[1]
    input_window_samples = X_train.shape[2]
    sfreq = config['sampling_rate']
    
    if model_type == 'EEGNetv4':
        """
        params = config.get('model_params', {}).get('EEGNetv4', {})
        model = EEGNetv4(n_chans=n_channels, n_outputs=n_classes, n_times=input_window_samples, sfreq=sfreq, **params).to(device)
        """
        model = EEGNetv4(in_chans=n_channels, n_classes=n_classes, input_window_samples=input_window_samples, sfreq=sfreq).to(device)
    elif model_type == 'ContraNet':
        """
        params = config.get('model_params', {}).get('ContraNet', {})
        if not params:
            warnings.warn("Configurazione per ContraNet non trovata in 'model_params'. Il modello userà i suoi valori di default se disponibili.")
        model = eegmodels.contranet(nb_classes=n_classes, Chans=n_channels, Samples=input_window_samples, **params).to(device)
        """
        model = eegmodels.contranet(n_classes, n_channels, input_window_samples, 0.5, 0.25, int(sfreq/2), 8, 16, 'SpatialDropout2D', 32, 1, 8, [64,32], [112]).to(device)
    elif model_type == 'EEGDeformer':
        """
        params = config.get('model_params', {}).get('EEGDeformer', {})
        model = eegmodels.eegdeformer(num_chan=n_channels, num_time=input_window_samples, num_classes=n_classes, **params).to(device)
        """
        model = eegmodels.eegdeformer(n_channels,input_window_samples, int(sfreq/10),num_classes=n_classes).to(device)    
    elif model_type == 'Conformer':
        
        params = config.get('model_params', {}).get('Conformer', {})
        patch_size, pool_size, pool_stride, emb_size = params.get('patch_size', 25), params.get('pool_size', 75), params.get('pool_stride', 15), params.get('emb_size', 40)
        length_after_conv = input_window_samples - patch_size + 1
        w = (length_after_conv - pool_size) // pool_stride + 1
        if w <= 0: raise ValueError(f"Conformer: Numero di patch calcolato (w={w}) non valido.")
        calculated_n_hidden = w * emb_size
        model = eegmodels.conformer(n_chan=n_channels, n_classes=n_classes, n_times=input_window_samples, custom_n_hidden=calculated_n_hidden, **params).to(device)
        
        #model = eegmodels.conformer(n_classes=n_classes, n_chan=n_channels, n_times=input_window_samples, n_patches=8,n_hidden=1040).to(device)
    elif model_type == 'EEGViT':
        """
        params = config.get('model_params', {}).get('EEGViT', {})
        model = eegmodels.eegvit(num_chan=n_channels, num_time=input_window_samples, num_classes=n_classes, **params).to(device)
        """
        model = eegmodels.eegvit(n_channels,input_window_samples,8, n_classes).to(device)
    elif model_type == 'ERTNet':
        params = config.get('model_params', {}).get('ERTNet', {})
        model = eegmodels.ertnet(nb_classes=n_classes, Chans=n_channels, Samples=input_window_samples, **params).to(device)
    elif model_type == 'EEGNet_custom':
        params = config.get('model_params', {}).get('EEGNet_custom', {})
        model = eegmodels.eegnet(nChan=n_channels, nTime=input_window_samples, nClass=n_classes, **params).to(device)
    else:
        raise ValueError(f"Tipo di modello '{model_type}' non supportato.")

    for param in model.parameters():
        param.requires_grad = True

    # --- 6. Definizione di Loss, Optimizer e Scheduler ---
    criterion_name = config.get('criterion_name', 'CrossEntropyLoss')
    optimizer_name = config.get('optimizer_name', 'Adam')
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
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    print(f"Inizio training per fold {fold_id_desc} su {device} | Classificazione: {is_classification}")

    batch_size = config['batch_size']
    num_epochs = config['epochs']
    
    for epoch in range(num_epochs):
        model.train()
        total_training_loss = 0
        train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), y_train), batch_size=batch_size, shuffle=True)
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_train = model(inputs)
            loss = criterion(outputs_train, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item()
        
        avg_training_loss = total_training_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        history['train_loss'].append(avg_training_loss)

        model.eval()
        total_val_loss, val_correct, val_total = 0.0, 0, 0
        val_loader = DataLoader(TensorDataset(torch.Tensor(X_val), y_val), batch_size=batch_size)
        y_val_indices = torch.LongTensor(y_val_orig).to(device)

        with torch.no_grad():
            all_val_preds = []
            all_val_targets = []
            for inputs, targets_one_hot_val in val_loader:
                inputs, targets_one_hot_val = inputs.to(device), targets_one_hot_val.to(device)
                outputs_val = model(inputs)
                loss = criterion(outputs_val, targets_one_hot_val)
                total_val_loss += loss.item()
                
                _, predicted_indices = torch.max(outputs_val.data, 1)
                all_val_preds.append(predicted_indices)
                
                _, targets_indices_batch = torch.max(targets_one_hot_val, 1)
                all_val_targets.append(targets_indices_batch)

            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            history['val_loss'].append(avg_val_loss)
            
            if all_val_preds:
                all_val_preds = torch.cat(all_val_preds)
                all_val_targets = torch.cat(all_val_targets)
                val_correct = (all_val_preds == all_val_targets).sum().item()
                val_total = all_val_targets.size(0)
                accuracy = val_correct / val_total if val_total > 0 else 0.0
            else:
                accuracy = 0.0
            history['val_accuracy'].append(accuracy)

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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_training_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val Acc: {accuracy * 100:.2f}%")

        if writer:
            writer.add_scalar('Loss/Train', avg_training_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            if is_classification:
                writer.add_scalar('Accuracy/Validation', accuracy, epoch)
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
        test_loader = DataLoader(TensorDataset(torch.Tensor(X_test)), batch_size=1)
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs_test = model(inputs)
                test_y_pred_list.append(torch.argmax(outputs_test, dim=1).cpu().numpy().flatten())
                
        y_pred = np.concatenate(test_y_pred_list) if test_y_pred_list else np.array([])
        y_test_indices = y_test[:len(y_pred)]

        if y_test_indices.size == 0:
            print("  ATTENZIONE: Nessuna predizione generata per il test set. Metriche finali non calcolate.")
            return {}, history

        final_metrics = {}
        final_metrics['accuracy'] = accuracy_score(y_test_indices, y_pred)
        final_metrics['f1_score'] = f1_score(y_test_indices, y_pred, average='weighted', zero_division=0)
        final_metrics['precision'] = precision_score(y_test_indices, y_pred, average='weighted', zero_division=0)
        final_metrics['recall'] = recall_score(y_test_indices, y_pred, average='weighted', zero_division=0)
        final_metrics['report'] = classification_report(y_test_indices, y_pred, zero_division=0, output_dict=True)
        final_metrics['conf_matrix_data'] = (y_test_indices, y_pred)

        print(final_metrics)

        if writer and final_metrics:
            hparams = {
                'lr': config['learning_rate'],
                'batch_size': config['batch_size'],
                'model_type': config['model_type'],
                'scaling': str(apply_scaling)
            }
            final_metrics_for_hparams = {
                'hparam/accuracy': final_metrics.get('accuracy', 0),
                'hparam/f1_score': final_metrics.get('f1_score', 0)
            }
            writer.add_hparams(hparams, final_metrics_for_hparams)

        return final_metrics, history

    return {}, history # Se skip_test_evaluation è True, restituisce metriche vuote