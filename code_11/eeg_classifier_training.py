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
    Esegue il training, la validazione e il test di un modello,
    con una logica differenziata per classificazione binaria e multi-classe.
    """
    # --- 1. Preparazione Dati e Parametri ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_classification = config['is_classification']
    apply_scaling = config.get('apply_scaling', True)
    model_type = config['model_type']
    fold_id_desc = config.get('fold_id')

    X_train, y_train_orig = data_splits['X_train'], data_splits['y_train']
    X_val, y_val_orig = data_splits['X_val'], data_splits['y_val']
    X_test, y_test = data_splits['X_test'], data_splits['y_test']

    # --- 4. Scaling Dati ---
    if apply_scaling and X_train.shape[0] > 0:
        scaler = mne.decoding.Scaler(scalings='mean')
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        if X_val.shape[0] > 0:
            X_val = scaler.transform(X_val)
        if X_test.shape[0] > 0:
            X_test = scaler.transform(X_test)

    # --- Calcolo robusto del numero di classi ---
    all_labels_in_fold = np.concatenate([arr for arr in [y_train_orig, y_val_orig, y_test] if arr.size > 0])
    n_classes_data = len(np.unique(all_labels_in_fold)) if all_labels_in_fold.size > 0 else 0
    
    # Decide il numero di neuroni in output e il formato delle etichette
    is_binary_classification = is_classification and n_classes_data == 2
    
    if is_binary_classification or is_classification: # Multi-classe (3, 4, etc.)
        print(f"  Rilevata classificazione BINARIA o MULTI-CLASSE ({n_classes_data} classi). Uso {n_classes_data} neuroni e one-hot encoding.")
        n_outputs = n_classes_data
        y_train = torch.nn.functional.one_hot(torch.LongTensor(y_train_orig), n_outputs).float()
        y_val = torch.nn.functional.one_hot(torch.LongTensor(y_val_orig), n_outputs).float()
    else: # Regressione
        print(f"  Rilevata REGRESSIONE. Uso 1 neurone in output.")
        n_outputs = 1
        y_train = torch.FloatTensor(y_train_orig).unsqueeze(1)
        y_val = torch.FloatTensor(y_val_orig).unsqueeze(1)
        
    # --- 5. Definizione del Modello ---
    n_channels = X_train.shape[1]
    input_window_samples = X_train.shape[2]
    
    # Qui inserisci la logica di selezione del modello come nel tuo file originale
    # (EEGNetv4, ContraNet, etc.). Assicurati di usare `n_outputs` come numero di output.
    if model_type == 'EEGNetv4':
        params = config.get('model_params', {}).get('EEGNetv4', {})
        model = EEGNetv4(n_chans=n_channels, n_outputs=n_outputs, n_times=input_window_samples, **params).to(device)
    # ... (aggiungi qui gli altri 'elif' per gli altri modelli, usando 'n_outputs')
    else:
        raise ValueError(f"Tipo di modello '{model_type}' non supportato.")

    for param in model.parameters():
        param.requires_grad = True

    # --- 6. Definizione di Loss, Optimizer e Scheduler ---
    criterion_name = config.get('criterion_name', 'L1Loss')
    optimizer_name = config.get('optimizer_name', 'Adam')
    criterion = getattr(nn, criterion_name)()
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config['learning_rate'], **config.get('optimizer_params', {}).get(optimizer_name, {}))
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config['early_stopping_patience'], factor=config['factor_scheduler'])

    # --- 7. Loop di Training e Validazione ---
    best_val_loss = float('inf')
    epoch_since_best = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    print(f"Inizio training per fold {fold_id_desc} su {device}")

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
        total_val_loss = 0.0
        all_val_preds_raw = []
        val_loader = DataLoader(TensorDataset(torch.Tensor(X_val), y_val), batch_size=batch_size)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs_val = model(inputs)
                loss = criterion(outputs_val, targets)
                total_val_loss += loss.item()
                all_val_preds_raw.append(outputs_val.cpu())

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        history['val_loss'].append(avg_val_loss)
        
        # Calcolo metriche di validazione
        if all_val_preds_raw:
            all_val_preds_raw = torch.cat(all_val_preds_raw).numpy()
            
            if is_binary_classification or is_classification: # Multi-classe
                preds = np.argmax(all_val_preds_raw, axis=1)
                accuracy = accuracy_score(y_val_orig, preds)
            else: # Regressione
                accuracy = r2_score(y_val_orig, all_val_preds_raw.flatten()) # R2 per regressione
            history['val_metrics'].append(accuracy)
        else:
            accuracy = 0.0
            history['val_metrics'].append(accuracy)

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
        
        metric_name = "Val Acc" if is_classification else "Val R2"
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_training_loss:.6f}, Val Loss: {avg_val_loss:.6f}, {metric_name}: {accuracy:.4f}")



    if not skip_test_evaluation:
        # --- 8. Calcolo delle Metriche Finali sul Test Set ---
        if X_test.shape[0] == 0:
            return {}, history

        model.load_state_dict(best_model_weights)
        model.eval()
        test_y_pred_raw = []
        test_loader = DataLoader(TensorDataset(torch.Tensor(X_test)), batch_size=batch_size)
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs_test = model(inputs)
                test_y_pred_raw.append(outputs_test.cpu())
        
        y_pred_raw = torch.cat(test_y_pred_raw).numpy()
        
        if is_binary_classification or is_classification: # Multi-classe
            y_pred = np.argmax(y_pred_raw, axis=1)
        else: # Regressione
            y_pred = y_pred_raw.flatten()

        y_test_final = y_test[:len(y_pred)]
        
        final_metrics = {}
        if is_classification:
            final_metrics['accuracy'] = accuracy_score(y_test_final, y_pred)
            final_metrics['f1_score'] = f1_score(y_test_final, y_pred, average='weighted', zero_division=0)
            final_metrics['precision'] = precision_score(y_test_final, y_pred, average='weighted', zero_division=0)
            final_metrics['recall'] = recall_score(y_test_final, y_pred, average='weighted', zero_division=0)
            final_metrics['report'] = classification_report(y_test_final, y_pred, zero_division=0, output_dict=True)
            final_metrics['conf_matrix_data'] = (y_test_final, y_pred)
        else: # Regressione
            final_metrics['r2_score'] = r2_score(y_test_final, y_pred)
            final_metrics['mse'] = mean_squared_error(y_test_final, y_pred)
            final_metrics['mae'] = mean_absolute_error(y_test_final, y_pred)
        
        print("Metriche finali sul test set:")
        print(final_metrics)
        return final_metrics, history

    return {}, history