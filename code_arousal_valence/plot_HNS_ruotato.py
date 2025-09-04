import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_hns_labels(dataset: str, label_type: str) -> np.ndarray:
    """
    Carica le etichette numeriche (0,1,2), le mappa a categoriali (H,S,N) e concatena.
    """
    label_type = label_type.upper()
    suffix_map = {'PUBLIC': '_hns_pubblico.npy', 'PRIVATE': '_discrete_emotions.npy'}
    if label_type not in suffix_map:
        logging.error(f"Tipo di etichetta non valido: {label_type}")
        return np.array([])
    data_dir = os.path.join('.', 'processed_data', f"{dataset.upper()}_NPY", 'LABEL', label_type)
    suffix = suffix_map[label_type]
    all_labels = []
    for i in range(8, 29):
        file_path = os.path.join(data_dir, f"P{i:03d}{suffix}")
        if os.path.isfile(file_path):
            try:
                all_labels.append(np.load(file_path, allow_pickle=True))
            except Exception as e:
                logging.warning(f"Errore caricamento {file_path}: {e}")
    if not all_labels:
        logging.warning(f"Nessun file trovato in {data_dir}")
        return np.array([])
    concat = np.concatenate(all_labels)
    label_map = {0: 'H', 1: 'S', 2: 'N'}
    mapped = np.array([label_map[x] for x in concat if x in label_map])
    logging.info(f"Caricati {len(all_labels)} soggetti da {data_dir}")
    return mapped


def calculate_transitions(pub: np.ndarray, priv: np.ndarray) -> dict:
    valid = {'H', 'S', 'N'}
    n = min(len(pub), len(priv))
    trans = defaultdict(int)
    for i in range(n):
        a, b = pub[i], priv[i]
        if a in valid and b in valid:
            trans[(a, b)] += 1
    return trans


def plot_hns_flow(name: str, public: np.ndarray, private: np.ndarray, output_dir: str):
    trans = calculate_transitions(public, private)
    total = sum(trans.values())
    if total == 0:
        logging.warning(f"Nessun dato valido per {name}")
        return

    cats = ['H', 'S', 'N']
    colors = {'H': '#2ca02c', 'S': '#d62728', 'N': '#7f7f7f'}  # più distinti
    base_angles = {'H': 0, 'S': 120, 'N': 240}

    # Calcola cerchi interni
    pub_totals = {c: sum(cnt for (a,_), cnt in trans.items() if a == c) for c in cats}
    radii = {}
    centers = {}
    for c in cats:
        cnt = trans.get((c, c), 0)
        ratio = cnt / pub_totals[c] if pub_totals[c] > 0 else 0
        r = np.sqrt(ratio) * 0.2
        theta = np.deg2rad(base_angles[c] + 60)
        centers[c] = (0.8 * np.cos(theta), 0.8 * np.sin(theta))
        radii[c] = r

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    # Disegna settori
    for c in cats:
        start_ang = base_angles[c]
        wedge = patches.Wedge((0,0), 1.0, start_ang, start_ang+120, facecolor=colors[c], alpha=0.15)
        ax.add_patch(wedge)
        mid_ang = np.deg2rad(start_ang + 60)
        ax.text(1.25*np.cos(mid_ang), 1.25*np.sin(mid_ang), c,
                ha='center', va='center', fontsize=24, fontweight='bold', color=colors[c])

    # Disegna frecce di transizione
    for (s,e), cnt in trans.items():
        if s == e or cnt == 0:
            continue
        c0 = np.array(centers[s]); c1 = np.array(centers[e])
        vec = c1 - c0
        dist = np.linalg.norm(vec)
        if dist == 0: continue
        unit = vec / dist
        start_pt = c0 + unit * radii[s]
        end_pt = c1 - unit * radii[e]
        lw = max(1, (cnt/total)*15)
        arrow = FancyArrowPatch(start_pt, end_pt,
                                arrowstyle='->', mutation_scale=15,
                                linewidth=lw, color=colors[s], alpha=0.8,
                                connectionstyle=f"arc3,rad={0.2 if cats.index(s)<cats.index(e) else -0.2}")
        ax.add_patch(arrow)
        mid = (start_pt+end_pt)/2
        ax.text(mid[0], mid[1], f"{cnt/total:.0%}", ha='center', va='center', fontsize=10)

    # Disegna cerchi di persistenza
    for c in cats:
        center = centers[c]; r = radii[c]
        circle = Circle(center, r, facecolor=colors[c], edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.add_patch(circle)
        if r > 0.05:
            ax.text(center[0], center[1], f"{trans.get((c,c),0)/pub_totals[c]:.0%}",
                    ha='center', va='center', color='white', fontsize=9)

    ax.set_title(f"Diagramma Flusso Emozionale: Pubblico vs Privato ({name})", fontsize=16)

    # Legenda posizionata a destra
    legend_elems = [Line2D([0],[0], color=colors[c], lw=4, label=c) for c in cats]
    ax.legend(handles=legend_elems, loc='upper left', bbox_to_anchor=(1.05,1), title='Categorie')

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name.lower().replace(' ','_')}_hns_flow.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    logging.info(f"Diagramma salvato in {path}")
    plt.show()


if __name__ == '__main__':
    output_dir = 'plots'
    # DEAP
    pub = load_hns_labels('deap','PUBLIC'); priv = load_hns_labels('deap','PRIVATE')
    if pub.size and priv.size:
        plot_hns_flow('DEAP',pub,priv,output_dir)
    # ImagEEG
    pub = load_hns_labels('graz','PUBLIC'); priv = load_hns_labels('graz','PRIVATE')
    if pub.size and priv.size:
        plot_hns_flow('ImagEEG (Graz)',pub,priv,output_dir)
    logging.info('Elaborazione completata.')
