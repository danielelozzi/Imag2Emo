import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_graz_data(folder_path="./label_private_filtered_p008_plus"):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in: {folder_path}")
    df_list = [pd.read_csv(os.path.join(folder_path, f)) for f in csv_files]
    if not df_list:
        raise ValueError("No data loaded; check your CSV files.")
    combined = pd.concat(df_list, ignore_index=True)
    mapping = {
        "valence_pubblico":    "public_valence",
        "arousal_pubblico":    "public_arousal",
        "rate_valence_privata":"private_valence",
        "rate_arousal_privato":"private_arousal",
    }
    if not all(k in combined.columns for k in mapping):
        raise ValueError("Missing required columns in CSV.")
    return combined[list(mapping)].rename(columns=mapping)

def get_quadrant(v, a, cx=5, cy=5):
    if   v>=cx and a>=cy: return 1
    elif v< cx and a>=cy: return 2
    elif v< cx and a< cy: return 3
    else:                 return 4

# 1) load & prep
df = load_graz_data()
df["pub_q"]  = df.apply(lambda r: get_quadrant(r.public_valence,   r.public_arousal), axis=1).astype(int)
df["priv_q"] = df.apply(lambda r: get_quadrant(r.private_valence,  r.private_arousal), axis=1).astype(int)

# 2) transition matrix
grouped = df.groupby(["pub_q","priv_q"]).size().reset_index(name="count")
T = np.zeros((4,4), dtype=int)
for _, row in grouped.iterrows():
    i,j = int(row["pub_q"])-1, int(row["priv_q"])-1
    T[i,j] = int(row["count"])

# 3) per-row totals
row_totals = T.sum(axis=1)

# 4) public & private counts per quadrant
pub_counts  = {q: df["pub_q"].value_counts().get(q,0)  for q in (1,2,3,4)}
priv_counts = {q: df["priv_q"].value_counts().get(q,0) for q in (1,2,3,4)}

# 5) plot setup
fig, ax = plt.subplots(figsize=(12,12))
ax.set(xlim=(1,10), ylim=(1,10),
       title="Emotion State Transitions (Public → Private)")
ax.axhline(5.5,color="k",lw=0.8); ax.axvline(5.5,color="k",lw=0.8)
ax.grid(True,linestyle="--",alpha=0.4)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])

# Aggiunta delle etichette degli assi all'interno del grafico
ax.text(5.5, 1.2, 'Valence', ha='center', va='center', fontsize=12, weight='bold', color='gray', alpha=0.7)
ax.text(1.2, 5.5, 'Arousal', ha='center', va='center', fontsize=12, weight='bold', rotation='vertical', color='gray', alpha=0.7)

quadrant_centers = {
    1:(7.75,7.75), 2:(3.25,7.75),
    3:(3.25,3.25), 4:(7.75,3.25)
}
colors   = plt.cm.tab20(np.linspace(0,1,16))
max_self = T.diagonal().max() or 1

# override positions
override_positions = {
    (4,3):(5.5,4.2), (3,4):(5.5,2.35),
    (3,2):(4.2,5.5), (2,3):(2.35,5.5),
    (2,4):(3.7,6.0), (3,1):(6.0,5.0),
    (2,1):(5.5,7.3), (4,1):(8.2,5.5),
    (4,2):(5.9,6.9),
}

# 6) draw circles & arrows
for i in range(4):
    for j in range(4):
        cnt = T[i,j]
        if not cnt: continue
        sq,eq = i+1,j+1
        color = colors[i*4+j]
        pct = cnt/row_totals[i]*100 if row_totals[i] else 0

        if sq==eq:
            cx,cy = quadrant_centers[sq]
            r = (cnt/max_self)*1.5
            ax.add_patch(patches.Circle((cx,cy),radius=r,color=color,alpha=0.6))
            ax.text(cx,cy,f"{pct:.1f}%",
                    ha="center",va="center",fontsize=12,
                    weight="bold",color="black")
        else:
            p=np.array(quadrant_centers[sq]); q=np.array(quadrant_centers[eq])
            rad=0.2 if T[j,i]==0 else 0.4
            ax.add_patch(patches.FancyArrowPatch(
                tuple(p),tuple(q),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",mutation_scale=15,
                lw=1+(cnt/df.shape[0])*15, color=color))
            if (sq,eq) in override_positions:
                lx,ly = override_positions[(sq,eq)]
            else:
                mid=(p+q)/2; vec=q-p
                perp=np.array([-vec[1],vec[0]])
                pu=perp/np.linalg.norm(perp)
                off=np.linalg.norm(vec)*0.1
                lx,ly = mid+pu*off*np.sign(rad)
            ax.text(lx,ly,f"{pct:.1f}%",ha="center",va="center",
                    fontsize=10,weight="bold",color=color,
                    bbox=dict(boxstyle="round,pad=0.2",fc="white",
                              ec=color,lw=1))

# 7) add quadrant legends
legend_positions = {
    1:(9.8,9.3),  # top-right
    2:(1.2,9.3),  # top-left
    3:(1.2,1.7),  # bottom-left
    4:(9.8,1.7),  # bottom-right
}
quadrant_names = {
    1: "HVHA", # High Valence, High Arousal
    2: "HALV", # High Arousal, Low Valence
    3: "LALV", # Low Arousal, Low Valence
    4: "LAHV"  # Low Arousal, High Valence
}
for q, (lx,ly) in legend_positions.items():
    quadrant_name = quadrant_names.get(q, "")
    txt = f"{quadrant_name}\nPublic: {pub_counts[q]}\nPrivate: {priv_counts[q]}"
    ax.text(lx, ly, txt,
            ha="left" if q in (2,3) else "right",
            va="top" if q in (1,2) else "bottom",
            fontsize=12, weight="normal",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))

# 8) save & show
plt.savefig("quadrant_transitions_quadrant_legends.png", dpi=300, bbox_inches="tight")
plt.show()
