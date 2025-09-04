import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_graz_data(folder_path="./label_private_filtered_p008_plus"):
    """
    Carica tutti i file CSV da una cartella specificata, li concatena e
    restituisce un DataFrame pandas con le colonne di valenza e arousal
    pubbliche e private.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"La cartella specificata non è stata trovata: '{os.path.abspath(folder_path)}'")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"Nessun file CSV trovato nella cartella: '{os.path.abspath(folder_path)}'")
    df_list = []
    for file in csv_files:
        try:
            df_list.append(pd.read_csv(os.path.join(folder_path, file)))
        except Exception as e:
            print(f"Errore nel leggere il file {file}: {e}")
    if not df_list:
        raise ValueError("Nessun dato è stato caricato. Controlla i file CSV.")
    combined_df = pd.concat(df_list, ignore_index=True)
    column_mapping = {
        'valence_pubblico': 'public_valence',
        'arousal_pubblico': 'public_arousal',
        'rate_valence_privata': 'private_valence',
        'rate_arousal_privato': 'private_arousal'
    }
    required_columns = list(column_mapping.keys())
    if not all(col in combined_df.columns for col in required_columns):
        raise ValueError("Una o più colonne richieste non sono state trovate nei file CSV.")
    final_df = combined_df[required_columns].rename(columns=column_mapping)
    print(f"Dati caricati e uniti da {len(csv_files)} file CSV. Totale campioni: {len(final_df)}.")
    return final_df

def get_quadrant(valence, arousal, center_x=5, center_y=5):
    """Restituisce il quadrante per una data coppia di valenza e arousal."""
    if valence >= center_x and arousal >= center_y:
        return 1 # HAHV
    elif valence < center_x and arousal >= center_y:
        return 2 # LAHV
    elif valence < center_x and arousal < center_y:
        return 3 # LALV
    else:
        return 4 # HALV

# --- Caricamento dati e calcolo transizioni ---
try:
    df = load_graz_data()
    print("Dati caricati con successo.")
except (FileNotFoundError, ValueError) as e:
    print(f"ERRORE: {e}. Esecuzione dello script interrotta.")
    exit()

df['public_quadrant']  = df.apply(lambda r: get_quadrant(r['public_valence'],  r['public_arousal']), axis=1)
df['private_quadrant'] = df.apply(lambda r: get_quadrant(r['private_valence'], r['private_arousal']), axis=1)

transitions = np.zeros((4, 4), dtype=int)
total_samples = len(df)
for _, row in df.iterrows():
    i = int(row['public_quadrant'])  - 1
    j = int(row['private_quadrant']) - 1
    transitions[i, j] += 1

# --- SEZIONE DI PLOTTING FINALE ---
fig, ax = plt.subplots(figsize=(12, 12))

# Impostazioni del grafico con titolo in inglese
center_val = 5.5
ax.axhline(center_val, color='black', linewidth=0.8)
ax.axvline(center_val, color='black', linewidth=0.8)
ax.set_xlim(1, 10)
ax.set_ylim(1, 10)
ax.set_xlabel('Valence', fontsize=14, weight='bold')
ax.set_ylabel('Arousal', fontsize=14, weight='bold')
ax.set_title('Transitions between Emotional States (Public -> Private)', fontsize=18, weight='bold')
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', alpha=0.5)

# Centri e nomi dei quadranti
quadrant_centers = {
    1: (7.75, 7.75), 2: (3.25, 7.75),
    3: (3.25, 3.25), 4: (7.75, 3.25),
}
quadrant_labels = {1: "HAHV", 2: "LAHV", 3: "LALV", 4: "HALV"}

# Aggiunge le etichette dei quadranti come sfondo trasparente
for q, (x, y) in quadrant_centers.items():
    ax.text(x, y, quadrant_labels[q], ha='center', va='center',
            fontsize=70, weight='bold', color='gray', alpha=0.15)


# Pre-calcolo degli offset per le frecce
offsets = np.zeros((4, 4, 2))
separation_distance = 1.0

for i in range(4):
    for j in range(i + 1, 4):
        if transitions[i, j] > 0 and transitions[j, i] > 0:
            start_pos = np.array(quadrant_centers[i + 1])
            end_pos   = np.array(quadrant_centers[j + 1])
            vec = end_pos - start_pos
            perp_vec = np.array([-vec[1], vec[0]])
            norm_perp_vec = perp_vec / np.linalg.norm(perp_vec)
            offsets[i, j] = norm_perp_vec * (separation_distance / 2)
            offsets[j, i] = -norm_perp_vec * (separation_distance / 2)

# Mappa di colori
colors = plt.cm.tab20(np.linspace(0, 1, 16))

# Disegna cerchi e frecce di transizione
for i in range(4):
    for j in range(4):
        count = transitions[i, j]
        if count == 0:
            continue

        start_q = i + 1
        end_q   = j + 1
        arrow_color = colors[i * 4 + j]
        
        # Calcola la percentuale e crea la stringa per l'etichetta
        percentage = (count / total_samples) * 100
        
        if start_q == end_q:
            # CERCHI CON ETICHETTA NUMERO + PERCENTUALE
            center = quadrant_centers[start_q]
            radius = np.log1p(count) * 0.2
            circle = patches.Circle(center, radius=radius, color=arrow_color, alpha=0.8)
            ax.add_patch(circle)
            
            # Etichetta su due righe per migliore leggibilità nel cerchio
            label_text = f"{int(count)}\n({percentage:.1f}%)"
            ax.text(center[0], center[1], label_text, ha='center', va='center',
                    fontsize=10, weight='bold', color='white', linespacing=1.4)

        else:
            # FRECCE CON ETICHETTA NUMERO + PERCENTUALE
            start_pos = np.array(quadrant_centers[start_q])
            end_pos   = np.array(quadrant_centers[end_q])
            arrow_width = 1 + (count / total_samples) * 20
            offset = offsets[i, j]
            final_start = start_pos + offset
            final_end = end_pos + offset

            arrow = patches.FancyArrowPatch(final_start, final_end,
                color=arrow_color, lw=arrow_width, arrowstyle="->", mutation_scale=30)
            ax.add_patch(arrow)

            label_pos = (final_start + final_end) / 2
            label_text = f"{int(count)} ({percentage:.1f}%)"
            ax.text(label_pos[0], label_pos[1], label_text, ha='center', va='center',
                    fontsize=9, weight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.15", fc='white', ec='none', alpha=0.8))

# Salvataggio e visualizzazione
plt.savefig("quadrant_transitions_final_en.png", dpi=300, bbox_inches='tight')
plt.show()