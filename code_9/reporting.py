# reporting.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

def plot_class_distribution(y_data, title, output_path):
    """
    Salva l'istogramma della distribuzione delle classi.
    """
    if y_data.ndim > 1: # Assicurati che sia 1D per np.bincount
        y_data = np.argmax(y_data, axis=1) # Se le label sono one-hot, prendi l'indice della classe

    class_counts = np.bincount(y_data.astype(int))
    classes = np.arange(len(class_counts))

    plt.figure(figsize=(8, 6))
    sns.barplot(x=classes, y=class_counts, palette='viridis')
    plt.title(f"Distribuzione Classi - {title}")
    plt.xlabel("Classe")
    plt.ylabel("Numero di Samples")
    plt.xticks(classes) # Assicura che le etichette dell'asse x siano i numeri delle classi
    plt.grid(axis='y', linestyle=':')
    plt.savefig(output_path)
    plt.close()

def plot_single_run_curves(history, title, output_path):
    """
    Salva il grafico con le curve di loss e accuracy per un singolo run,
    utilizzando un doppio asse Y.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Asse Y primario (a sinistra) per la LOSS ---
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], label='Training Loss', color='lightcoral', linestyle='--')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle=':')

    # --- Asse Y secondario (a destra) per l'ACCURACY ---
    # Controlla se i dati dell'accuracy esistono nel dizionario di history
    if 'val_accuracy' in history and history['val_accuracy']:
        ax2 = ax1.twinx()  # Crea un secondo asse che condivide l'asse x
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='blue')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1.05) # Fissa il limite per l'accuracy tra 0 e 1

    # --- Titolo e Legenda Unificata ---
    fig.suptitle(f"Training & Validation Curves - {title}")
    
    # Raccoglie le etichette da entrambi gli assi per creare una legenda unica
    lines, labels = ax1.get_legend_handles_labels()
    if 'val_accuracy' in history and history['val_accuracy']:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    
    ax1.legend(lines, labels, loc='upper right')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Aggiusta il layout per fare spazio al super-titolo
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title, output_path):
    """Salva la matrice di confusione."""
    if y_true.size == 0 or y_pred.size == 0:
        print(f"Attenzione: y_true o y_pred sono vuoti per {title}. Impossibile generare la matrice di confusione.")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Dati non sufficienti per la matrice di confusione", horizontalalignment='center', verticalalignment='center')
        plt.title(f"Confusion Matrix - {title} (Dati Insufficienti)")
        plt.savefig(output_path)
        plt.close()
        return

    # Se le classi non sono fornite, derivale dai dati per robustezza
    if classes is None:
        classes = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()


def save_classification_report(report_dict, output_path):
    """Salva il classification report in formato JSON."""
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=4)

def plot_average_curves(histories, metric, title, output_path):
    """Calcola e plotta le curve medie e la deviazione standard da più run."""
    plt.figure(figsize=(12, 7))
    
    all_runs = [h[metric] for h in histories if metric in h]
    if not all_runs:
        print(f"Attenzione: nessuna history trovata per la metrica '{metric}'. Grafico non generato.")
        plt.close()
        return

    max_len = max(len(run) for run in all_runs)
    padded_runs = [np.pad(run, (0, max_len - len(run)), 'constant', constant_values=np.nan) for run in all_runs]
    
    mean_curve = np.nanmean(padded_runs, axis=0)
    std_curve = np.nanstd(padded_runs, axis=0)
    
    epochs = np.arange(max_len)
    
    plt.plot(epochs, mean_curve, label=f'Mean {metric.replace("_", " ").title()}')
    plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, label=f'Std Dev')
    
    plt.title(f'Average {metric.replace("_", " ").title()} - {title}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_all_curves_in_one(histories, title, output_path):
    """Calcola e plotta tutte le curve (loss e accuracy) medie e deviazione standard in un singolo grafico."""
    metrics = ['train_loss', 'val_loss', 'val_accuracy']
    plt.figure(figsize=(12, 7))

    for metric in metrics:
        all_runs = [h[metric] for h in histories if metric in h]
        if all_runs:
            max_len = max(len(run) for run in all_runs)
            padded_runs = [np.pad(run, (0, max_len - len(run)), 'constant', constant_values=np.nan) for run in all_runs]
            mean_curve = np.nanmean(padded_runs, axis=0)
            std_curve = np.nanstd(padded_runs, axis=0)
            epochs = np.arange(max_len)
            plt.plot(epochs, mean_curve, label=f'Mean {metric.replace("_", " ").title()}')
            plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.title(f'Average Training and Validation Metrics - {title}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def save_summary_metrics(results_df, output_path, reports_list=None):
    """
    Salva il DataFrame con le metriche riassuntive e, se fornita una lista di report,
    calcola e salva anche un riepilogo delle metriche per classe.
    """
    if results_df.empty:
        print(f"Attenzione: DataFrame dei risultati vuoto. Nessun riepilogo salvato in '{output_path}'.")
        return
        
    # --- 1. Salva il riepilogo delle metriche generali (accuracy, f1-score globale, etc.) ---
    summary_stats = results_df.drop(columns=['fold']).agg(['mean', 'std']).transpose()
    results_df.loc['mean'] = results_df.mean(numeric_only=True)
    results_df.loc['std'] = results_df.std(numeric_only=True)
    
    results_df.to_csv(output_path)
    print(f"\n--- Riepilogo metriche generali salvato in: {output_path} ---")
    print("Statistiche generali (media e dev. std.):")
    print(summary_stats)

    # --- 2. Se presenti i report di classificazione, calcola e salva le medie per classe ---
    if reports_list:
        records = []
        for report in reports_list:
            for class_label, metrics in report.items():
                if isinstance(metrics, dict):
                    records.append({
                        'class': class_label,
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1-score': metrics['f1-score'],
                        'support': metrics['support']
                    })
        
        if not records:
            print("Nessun dato di classificazione trovato nei report per il riepilogo per classe.")
            return

        per_class_df = pd.DataFrame(records)
        summary_per_class = per_class_df.groupby('class')[['precision', 'recall', 'f1-score']].agg(['mean', 'std'])
        summary_per_class = summary_per_class.round(4)

        per_class_output_path = output_path.replace('.csv', '_per_classe.csv')
        summary_per_class.to_csv(per_class_output_path)

        print(f"\n--- Riepilogo metriche per classe salvato in: {per_class_output_path} ---")
        print("Statistiche medie per classe (precision, recall, f1-score):")
        print(summary_per_class)