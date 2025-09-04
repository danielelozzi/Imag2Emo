import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_graz_data(folder_path="./label_private_filtered_p008_plus"):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Cartella non trovata: {folder_path}")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"Nessun CSV in: {folder_path}")
    df_list = []
    for fname in csv_files:
        path = os.path.join(folder_path, fname)
        try:
            df_list.append(pd.read_csv(path))
        except Exception as e:
            print(f"Errore leggendo {fname}: {e}")
    if not df_list:
        raise ValueError("Nessun dato caricato, controlla i file CSV.")
    combined = pd.concat(df_list, ignore_index=True)
    mapping = {
        "valence_pubblico": "public_valence",
        "arousal_pubblico": "public_arousal",
        "rate_valence_privata": "private_valence",
        "rate_arousal_privato": "private_arousal",
    }
    if not all(k in combined.columns for k in mapping):
        raise ValueError("Mancano colonne richieste nei CSV.")
    return combined[list(mapping)].rename(columns=mapping)

def get_quadrant(v, a, cx=5, cy=5):
    if   v >= cx and a >= cy: return 1
    elif v <  cx and a >= cy: return 2
    elif v <  cx and a <  cy: return 3
    else:                     return 4

# 1) Carico e preparo i dati
df = load_graz_data()
df["pub_q"]  = df.apply(lambda r: get_quadrant(r.public_valence,  r.public_arousal), axis=1).astype(int)
df["priv_q"] = df.apply(lambda r: get_quadrant(r.private_valence, r.private_arousal), axis=1).astype(int)

# 2) Costruisco la matrice di transizione
grouped = df.groupby(["pub_q","priv_q"]).size().reset_index(name="count")
T = np.zeros((4,4), dtype=int)
for _, row in grouped.iterrows():
    i, j = int(row["pub_q"])-1, int(row["priv_q"])-1
    T[i,j] = int(row["count"])

# 3) Setup del plot
fig, ax = plt.subplots(figsize=(12,12))
ax.set(xlim=(1,10), ylim=(1,10),
       xlabel="Valence", ylabel="Arousal",
       title="Label transition form Public to Private (ImagEEG)")
ax.axhline(5.5, color="k", lw=0.8)
ax.axvline(5.5, color="k", lw=0.8)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_aspect("equal")

quadrant_centers = {
    1: (7.75,7.75),
    2: (3.25,7.75),
    3: (3.25,3.25),
    4: (7.75,3.25),
}
colors   = plt.cm.tab20(np.linspace(0,1,16))
max_self = T.diagonal().max() or 1

# 4) Override manuale delle posizioni per le etichette specifiche
override_positions = {
    (4,3): (5.5, 4.2),  # 330
    (3,4): (5.5, 2.35),  # 147
    (3,2): (4.2, 5.5),  # 214
    (2,3): (2.35, 5.5),  # 35
    (2,4): (3.7, 6.0),  # 20
    (3,1): (6.0, 5.0),  # 83
    (2,1): (5.5, 7.3),  # 16
    (4,1): (8.2, 5.5),  # 395
    (4,2): (5.9, 6.9),  # 77
}

# 5) Disegno cerchi e frecce con etichette
for i in range(4):
    for j in range(4):
        cnt = T[i,j]
        if cnt == 0:
            continue

        sq, eq = i+1, j+1
        color = colors[i*4 + j]
        lw    = 1 + (cnt/df.shape[0])*15

        if sq == eq:
            # self-transition: cerchio proporzionale
            cx, cy = quadrant_centers[sq]
            r = (cnt/max_self)*1.5
            ax.add_patch(patches.Circle((cx,cy), radius=r, color=color, alpha=0.6))
            ax.text(cx, cy, str(cnt),
                    ha="center", va="center",
                    fontsize=12, weight="bold", color="black")
        else:
            # freccia curva
            p = np.array(quadrant_centers[sq])
            q = np.array(quadrant_centers[eq])
            rad = 0.2 if T[j,i] == 0 else 0.4

            ax.add_patch(patches.FancyArrowPatch(
                tuple(p), tuple(q),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=15,
                lw=lw, color=color
            ))

            # Determino posizione dell'etichetta
            if (sq,eq) in override_positions:
                label_x, label_y = override_positions[(sq,eq)]
            else:
                # fallback: punto medio con offset dinamico
                mid = (p+q)/2
                vec = q - p
                perp = np.array([-vec[1], vec[0]])
                perp_unit = perp / np.linalg.norm(perp)
                offset = np.linalg.norm(vec)*0.1
                label_x, label_y = (mid + perp_unit*offset*np.sign(rad))

            ax.text(label_x, label_y, str(cnt),
                    ha="center", va="center",
                    fontsize=10, weight="bold",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec=color, lw=1))

# 6) Salvo e mostro
plt.savefig("quadrant_transitions_custom_labels.png", dpi=300, bbox_inches="tight")
plt.show()
