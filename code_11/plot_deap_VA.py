import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_deap_data(base_folder="./LABEL_DEAP"):
    """
    Carica i dati DEAP dai file .npy nelle sottocartelle PUBLIC e PRIVATE.
    """
    public_folder = os.path.join(base_folder, "PUBLIC")
    private_folder = os.path.join(base_folder, "PRIVATE")

    if not os.path.isdir(public_folder) or not os.path.isdir(private_folder):
        raise FileNotFoundError(f"Una o entrambe le cartelle non sono state trovate: {public_folder}, {private_folder}")

    # Ottiene una lista di tutti i file .npy e estrae gli ID dei soggetti
    public_files = [f for f in os.listdir(public_folder) if f.endswith(".npy")]
    private_files = [f for f in os.listdir(private_folder) if f.endswith(".npy")]

    all_files = public_files + private_files
    subjects = sorted(list(set([f.split('_')[0] for f in all_files])))

    if not subjects:
        raise FileNotFoundError(f"Nessun file .npy trovato in: {public_folder} o {private_folder}")

    df_list = []
    for subject in subjects:
        # Costruisce i nomi dei file per il soggetto corrente
        valence_pub_path = os.path.join(public_folder, f"{subject}_valence_pubblica.npy")
        arousal_pub_path = os.path.join(public_folder, f"{subject}_arousal_pubblica.npy")
        valence_priv_path = os.path.join(private_folder, f"{subject}_valence_privata.npy")
        arousal_priv_path = os.path.join(private_folder, f"{subject}_arousal_privata.npy")

        # Carica i file .npy
        try:
            valence_pub = np.load(valence_pub_path)
            arousal_pub = np.load(arousal_pub_path)
            valence_priv = np.load(valence_priv_path)
            arousal_priv = np.load(arousal_priv_path)

            # Crea un DataFrame per il soggetto
            temp_df = pd.DataFrame({
                "valence_pubblica": valence_pub,
                "arousal_pubblica": arousal_pub,
                "valence_privata": valence_priv,
                "arousal_privata": arousal_priv
            })
            df_list.append(temp_df)
        except FileNotFoundError as e:
            print(f"Attenzione: File mancante per il soggetto {subject}. Salto. ({e})")
            continue

    if not df_list:
        raise ValueError("Nessun dato caricato; controlla i tuoi file .npy.")

    combined = pd.concat(df_list, ignore_index=True)

    # Rinomina le colonne per corrispondere al resto dello script
    mapping = {
        "valence_pubblica": "public_valence",
        "arousal_pubblica": "public_arousal",
        "valence_privata": "private_valence",
        "arousal_privata": "private_arousal",
    }
    return combined[list(mapping)].rename(columns=mapping)

def plot_valence_transitions(df):
    df["pv"] = (df.public_valence >= 4.5).astype(int)
    df["prv"] = (df.private_valence >= 4.5).astype(int)

    ct = pd.crosstab(df.pv, df.prv) \
            .reindex(index=[0,1], columns=[0,1]) \
            .fillna(0).astype(int)
    M = ct.to_numpy()
    row_totals = M.sum(axis=1)
    total = len(df)
    max_self = M.diagonal().max() or 1

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=(1,10), ylim=(1,10),
            title="Transiction from Public → Private - VALENCE")
    ax.axvline(5.5, color="k", lw=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    centers = {0:(3.25,5.5), 1:(7.75,5.5)}
    colors = plt.cm.tab20(np.linspace(0,1,4))

    # cerchi auto-transizione
    for state in (0,1):
        c = M[state,state]
        if c>0:
            pct = c / row_totals[state] * 100
            x,y = centers[state]
            r = (np.log1p(c) / np.log1p(max_self)) * 1.5
            ax.add_patch(patches.Circle((x,y), r, color=colors[state*2+state], alpha=0.6))
            ax.text(x, y, f"{pct:.1f}%", ha="center", va="center", weight="bold")

    # frecce cross-state con un filo di curvatura e offset verticale
    for i,j in [(0,1),(1,0)]:
        c = M[i,j]
        if c==0: continue
        pct = c / row_totals[i] * 100
        p = np.array(centers[i])
        q = np.array(centers[j])
        dy = 0.7 if i<j else -0.7
        start = p + np.array([0, dy])
        end = q + np.array([0, dy])

        # rad = -0.2 per 0→1, -0.2 per 1→0
        rad = -0.2 if i<j else -0.2

        arrow = patches.FancyArrowPatch(
            start, end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=15,
            lw=1 + (c/total)*15,
            color=colors[i*2+j],
            shrinkA=10, shrinkB=10
        )
        ax.add_patch(arrow)

        mid = (start + end) / 2
        ax.text(mid[0], mid[1], f"{pct:.1f}%",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2",
                            fc="white", ec=colors[i*2+j], lw=1),
                weight="bold")

    # etichette stati
    pub = df.pv.value_counts().sort_index()
    priv = df.prv.value_counts().sort_index()
    ax.text(1.2,9.3,
            f"LV\nPublic: {pub.get(0,0)}\nPrivate: {priv.get(0,0)}",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="black"))
    ax.text(9.8,9.3,
            f"HV\nPublic: {pub.get(1,0)}\nPrivate: {priv.get(1,0)}",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="black"))

    plt.savefig("valence_transitions_deap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Salvato: valence_transitions_deap.png")

def plot_arousal_transitions(df):
    
    df["pa"] = np.round(df.public_arousal , 0).astype(int)
    df["pra"] = np.round(df.private_arousal, 0).astype(int)

    df["pa"]  = (df.public_arousal  >= 5).astype(int)
    df["pra"] = (df.private_arousal >= 5).astype(int)

    ct = pd.crosstab(df.pa, df.pra) \
            .reindex(index=[0,1], columns=[0,1]) \
            .fillna(0).astype(int)
    M = ct.to_numpy()
    row_totals = M.sum(axis=1)
    total = len(df)
    max_self = M.diagonal().max() or 1

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlim=(1,10), ylim=(1,10),
            title="Transiction from Public → Private - AROUSAL")
    ax.axhline(5.5, color="k", lw=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    centers = {0:(5.5,3.25), 1:(5.5,7.75)}
    colors = plt.cm.tab20(np.linspace(0,1,4))

    # cerchi auto-transizione
    for state in (0,1):
        c = M[state,state]
        if c>0:
            pct = c / row_totals[state] * 100
            x,y = centers[state]
            r = (np.log1p(c) / np.log1p(max_self)) * 1.5
            ax.add_patch(patches.Circle((x,y), r, color=colors[state*2+state], alpha=0.6))
            ax.text(x, y, f"{pct:.1f}%", ha="center", va="center", weight="bold")

    # frecce cross-state con un filo di curvatura e offset orizzontale
    for i,j in [(0,1),(1,0)]:
        c = M[i,j]
        if c==0: continue
        pct = c / row_totals[i] * 100
        p = np.array(centers[i])
        q = np.array(centers[j])
        dx = 0.7 if i<j else -0.7
        start = p + np.array([dx, 0])
        end = q + np.array([dx, 0])

        rad = +0.2 if i<j else +0.2

        arrow = patches.FancyArrowPatch(
            start, end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=15,
            lw=1 + (c/total)*15,
            color=colors[i*2+j],
            shrinkA=10, shrinkB=10
        )
        ax.add_patch(arrow)

        mid = (start + end) / 2
        ax.text(mid[0], mid[1], f"{pct:.1f}%",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2",
                            fc="white", ec=colors[i*2+j], lw=1),
                weight="bold")

    # etichette stati
    pub = df.pa.value_counts().sort_index()
    priv = df.pra.value_counts().sort_index()
    ax.text(1.2,1.7,
            f"LA\nPublic: {pub.get(0,0)}\nPrivate: {priv.get(0,0)}",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="black"))
    ax.text(1.2,9.3,
            f"HA\nPublic: {pub.get(1,0)}\nPrivate: {priv.get(1,0)}",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="black"))

    plt.savefig("arousal_transitions_deap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Salvato: arousal_transitions_deap.png")

if __name__ == "__main__":
    df = load_deap_data("./LABEL_DEAP")
    plot_valence_transitions(df.copy())
    plot_arousal_transitions(df.copy())