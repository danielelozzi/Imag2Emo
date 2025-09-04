import matplotlib.pyplot as plt# Carica i dati reali dal dataset GRAZ
import os
import numpy as np
import pandas as pd


def load_graz_data(folder_path="./label_private_filtered_p008_plus"):
    """
    Carica tutti i file CSV da una cartella specificata, li concatena e 
    restituisce un DataFrame pandas con le colonne di valenza e arousal 
    pubbliche e private.

    Args:
        folder_path (str): Il percorso della cartella contenente i file CSV.

    Returns:
        pd.DataFrame: Un DataFrame contenente le colonne:
                      'public_valence', 'public_arousal',
                      'private_valence', 'private_arousal'.
    
    Raises:
        FileNotFoundError: Se la cartella specificata non esiste o è vuota.
    """
    # Controlla se la cartella esiste
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"La cartella specificata non è stata trovata: '{os.path.abspath(folder_path)}'")

    # Trova tutti i file CSV nella cartella
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"Nessun file CSV trovato nella cartella: '{os.path.abspath(folder_path)}'")

    # Leggi e concatena tutti i file CSV in un unico DataFrame
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            df_list.append(pd.read_csv(file_path))
        except Exception as e:
            print(f"Errore nel leggere il file {file}: {e}")
    
    if not df_list:
        raise ValueError("Nessun dato è stato caricato. Controlla i file CSV.")

    combined_df = pd.concat(df_list, ignore_index=True)

    # Mappatura delle colonne originali ai nuovi nomi desiderati
    column_mapping = {
        'valence_pubblico': 'public_valence',
        'arousal_pubblico': 'public_arousal',
        'rate_valence_privata': 'private_valence',
        'rate_arousal_privato': 'private_arousal'
    }

    # Seleziona solo le colonne necessarie
    required_columns = list(column_mapping.keys())
    # Controlla se tutte le colonne necessarie sono presenti nel DataFrame
    if not all(col in combined_df.columns for col in required_columns):
        raise ValueError("Una o più colonne richieste non sono state trovate nei file CSV.")
        
    final_df = combined_df[required_columns]

    # Rinomina le colonne
    final_df = final_df.rename(columns=column_mapping)
    
    print(f"Dati caricati e uniti da {len(csv_files)} file CSV. Totale campioni: {len(final_df)}.")

    return final_df


def get_quadrant(valence, arousal, center_x=5, center_y=5):
    """Restituisce il quadrante per una data coppia di valenza e arousal."""
    if valence >= center_x and arousal >= center_y:
        return 1
    elif valence < center_x and arousal >= center_y:
        return 2
    elif valence < center_x and arousal < center_y:
        return 3
    elif valence >= center_x and arousal < center_y:
        return 4

try:
    df = load_graz_data()
    print("Dati caricati con successo.")
except FileNotFoundError as e:
    print(f"ERRORE: {e}. Esecuzione dello script interrotta.")
    # Esce dallo script se i dati non vengono trovati
    exit()

# Applica la funzione per ottenere i quadranti
df['public_quadrant'] = df.apply(lambda row: get_quadrant(row['public_valence'], row['public_arousal']), axis=1)
df['private_quadrant'] = df.apply(lambda row: get_quadrant(row['private_valence'], row['private_arousal']), axis=1)

# Calcola le transizioni
transitions = np.zeros((4, 4))
for _, row in df.iterrows():
    pub_quad = int(row['public_quadrant']) - 1
    priv_quad = int(row['private_quadrant']) - 1
    transitions[pub_quad, priv_quad] += 1

# --- Creazione del Grafico ---
fig, ax = plt.subplots(figsize=(12, 12))

# Impostazioni degli assi
center = 5.5
ax.axhline(center, color='black', linewidth=0.8)
ax.axvline(center, color='black', linewidth=0.8)
ax.set_xlim(1, 10)
ax.set_ylim(1, 10)
ax.set_xlabel('Valence')
ax.set_ylabel('Arousal')
ax.set_title('Relazione tra Label Pubbliche e Private (Spessore Variabile)')
ax.set_aspect('equal', adjustable='box')

quadrant_centers = {
    1: (7.75, 7.75),
    2: (3.25, 7.75),
    3: (3.25, 3.25),
    4: (7.75, 3.25),
}

for q_num, pos in quadrant_centers.items():
    ax.text(pos[0], pos[1] + 0.5, f'Quadrante {q_num}', ha='center', va='center', fontsize=12, weight='bold')

# Disegna cerchi e frecce
for i in range(4):
    for j in range(4):
        count = transitions[i, j]
        if count == 0:
            continue

        start_quad_num = i + 1
        end_quad_num = j + 1

        if start_quad_num == end_quad_num:
            # Stesso quadrante: disegna un cerchio
            center_x, center_y = quadrant_centers[start_quad_num]
            radius = np.sqrt(count) * 0.1
            circle = plt.Circle((center_x, center_y), radius, color='skyblue', alpha=0.7)
            ax.add_artist(circle)
            ax.text(center_x, center_y, str(int(count)), ha='center', va='center', fontsize=10, weight='bold')
        else:
            # Cambio di quadrante: disegna una freccia
            start_pos = np.array(quadrant_centers[start_quad_num])
            end_pos = np.array(quadrant_centers[end_quad_num])
            
            # --- MODIFICA: Aumentato il fattore per rendere lo spessore più evidente ---
            arrow_width = count * 0.01 # Puoi aumentare o diminuire questo valore
            
            # Disegna la freccia passando lo spessore calcolato (lw)
            ax.annotate("", xy=end_pos, xycoords='data', xytext=start_pos, textcoords='data',
                        arrowprops=dict(arrowstyle="->", color="tomato", connectionstyle="arc3,rad=0.1",
                                        shrinkA=15, shrinkB=15)) # lw=arrow_width
            
            # Calcolo della posizione della label
            direction_vec = end_pos - start_pos
            perp_vec = np.array([-direction_vec[1], direction_vec[0]])
            norm = np.linalg.norm(perp_vec)
            if norm > 0:
                perp_vec_unit = perp_vec / norm
            else:
                perp_vec_unit = np.array([0, 0])

            mid_point = (start_pos + end_pos) / 2
            offset_distance = 0.4 
            label_pos = mid_point + perp_vec_unit * offset_distance
            
            ax.text(label_pos[0], label_pos[1], str(int(count)), ha='center', va='center', fontsize=9, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.9))

plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("quadrant_transitions_thickness.png", dpi=300)
plt.show()

# ... (tutto il codice fino alla fine del primo plot)

# --- NUOVA SEZIONE: PLOT DI TUTTE LE TRANSIZIONI INDIVIDUALI ---

# Incolla qui la nuova funzione plot_all_transitions(df)
def plot_all_transitions(df):
    """
    Crea un grafico che mostra la transizione dalla valutazione pubblica a quella
    privata per OGNI campione nel DataFrame.
    ... (corpo della funzione come sopra)
    """
    # (corpo completo della funzione qui)
    fig, ax = plt.subplots(figsize=(12, 12))
    center = 5.5
    ax.axhline(center, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(center, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 10)
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_title(f'Transizioni da Pubblico a Privato (Tutti i {len(df)} Campioni)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.5)
    for index, row in df.iterrows():
        public_point = (row['public_valence'], row['public_arousal'])
        private_point = (row['private_valence'], row['private_arousal'])
        ax.plot(public_point[0], public_point[1], 'o', color='blue', markersize=5, alpha=0.6, label='Pubblico' if index == df.index[0] else "")
        ax.plot(private_point[0], private_point[1], 'o', color='red', markersize=5, alpha=0.6, label='Privato' if index == df.index[0] else "")
        ax.annotate("", xy=private_point, xycoords='data', xytext=public_point, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="gray", shrinkA=4, shrinkB=4, linestyle="--", linewidth=0.8, alpha=0.7))
    ax.legend()
    plt.savefig("all_transitions_plot.png", dpi=300)
    print(f"\nGrafico di tutte le {len(df)} transizioni salvato come 'all_transitions_plot.png'")
    plt.show()


# Chiama la nuova funzione dopo aver caricato i dati
if 'df' in locals() and not df.empty:
    plot_all_transitions(df)