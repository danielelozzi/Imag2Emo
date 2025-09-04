import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Le funzioni 'generate_dummy_data' e 'get_quadrant' rimangono invariate
def generate_dummy_data(num_images=100):
    """Genera un DataFrame con dati fittizi."""
    data = {
        'image_id': range(num_images),
        'public_valence': np.random.uniform(1, 9, num_images),
        'public_arousal': np.random.uniform(1, 9, num_images),
        'private_valence': np.random.uniform(1, 9, num_images),
        'private_arousal': np.random.uniform(1, 9, num_images),
    }
    return pd.DataFrame(data)

def get_quadrant(valence, arousal, center_x=5.5, center_y=5.5):
    """Restituisce il quadrante per una data coppia di valenza e arousal."""
    if valence >= center_x and arousal >= center_y:
        return 1
    elif valence < center_x and arousal >= center_y:
        return 2
    elif valence < center_x and arousal < center_y:
        return 3
    elif valence >= center_x and arousal < center_y:
        return 4

# Genera i dati
df = generate_dummy_data(200)

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
            arrow_width = count * 0.07 # Puoi aumentare o diminuire questo valore
            
            # Disegna la freccia passando lo spessore calcolato (lw)
            ax.annotate("", xy=end_pos, xycoords='data', xytext=start_pos, textcoords='data',
                        arrowprops=dict(arrowstyle="->", color="tomato", connectionstyle="arc3,rad=0.1",
                                        lw=arrow_width, shrinkA=15, shrinkB=15))
            
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