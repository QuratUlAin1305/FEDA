"""Stage 2: the four FEDA blocks under the corrected definitions.

FG   28   fundamental group of the molecular graph
ESPH 144  element-specific persistent homology
ADM  47   actual distance matrix (overflow bin, padded spectrum)
AM   42   angle matrix = bond angles at each central atom (trace dropped)
          -------
          261
"""
import sys, pickle, warnings
import numpy as np
import networkx as nx
from itertools import combinations

warnings.filterwarnings('ignore')

HALOGENS = {'F', 'Cl', 'Br', 'I', 'At'}
POLAR = {'N', 'O', 'S', 'P'}


# ---------------------------------------------------------------- helpers
def moments(v):
    """mean, std, skewness, kurtosis (Fisher)."""
    if len(v) == 0:
        return [0.0] * 4
    m, s = float(np.mean(v)), float(np.std(v))
    if s < 1e-12 or len(v) < 3:
        return [m, s, 0.0, 0.0]
    z = (v - m) / s
    return [m, s, float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)]


def order_stats(v):
    """min, max, median, q25, q75."""
    if len(v) == 0:
        return [0.0] * 5
    return [float(np.min(v)), float(np.max(v)), float(np.median(v)),
            float(np.percentile(v, 25)), float(np.percentile(v, 75))]


def pad_spectrum(lam, n_top, n_bot):
    """Eigenvalues sorted descending; take n_top from the top and n_bot from the
    bottom, zero-padding when the molecule is too small to supply a value."""
    lam = np.sort(np.asarray(lam, dtype=float))[::-1]
    top = list(lam[:n_top]) + [0.0] * max(0, n_top - len(lam))
    bot = list(lam[::-1][:n_bot]) + [0.0] * max(0, n_bot - len(lam))
    return [float(x) for x in top + bot]


def graph_of(d):
    G = nx.Graph()
    G.add_nodes_from(range(d['n']))
    G.add_edges_from(d['edges'])
    return G


# ---------------------------------------------------------------- FG block (28)
def fg_block(d):
    G = graph_of(d)
    n, m = G.number_of_nodes(), G.number_of_edges()
    c = nx.number_connected_components(G)
    r = m - n + c
    try:
        basis = nx.minimum_cycle_basis(G) if r > 0 else []
    except Exception:
        basis = []
    lens = np.array([len(b) for b in basis], dtype=float)

    f = [float(n), float(m), float(c), float(r), r / n if n else 0.0,
         float(len(basis))]
    if len(lens):
        f += [float(lens.min()), float(lens.max()), float(lens.mean()),
              float(lens.std()), float(lens.sum())]
    else:
        f += [0.0] * 5
    for L in (3, 4, 5, 6, 7):
        f.append(float(np.sum(lens == L)))
    f.append(float(np.sum(lens >= 8)))

    on_cycle = set().union(*basis) if basis else set()
    f += [float(len(on_cycle)), len(on_cycle) / n if n else 0.0]

    # atoms and edges carried by two or more representatives
    atom_ct, edge_ct = {}, {}
    for b in basis:
        sub = G.subgraph(b)
        cyc_edges = [tuple(sorted(e)) for e in sub.edges()]
        for a in b:
            atom_ct[a] = atom_ct.get(a, 0) + 1
        for e in cyc_edges:
            edge_ct[e] = edge_ct.get(e, 0) + 1
    f += [float(sum(1 for v in edge_ct.values() if v >= 2)),
          float(sum(1 for v in atom_ct.values() if v >= 2))]
    arom = d['arom']
    f.append(float(sum(1 for b in basis if all(arom[a] for a in b))))

    # induced-subgraph ranks
    sym = d['sym']
    sel = {
        'C':   [i for i, s in enumerate(sym) if s == 'C'],
        'NO':  [i for i, s in enumerate(sym) if s in ('N', 'O')],
        'CNO': [i for i, s in enumerate(sym) if s in ('C', 'N', 'O')],
        'X':   [i for i, s in enumerate(sym) if s in HALOGENS],
        'ar':  [i for i in range(d['n']) if arom[i]],
        'het': [i for i, s in enumerate(sym) if s != 'C'],
    }
    for key in ('C', 'NO', 'CNO', 'X', 'ar', 'het'):
        S = sel[key]
        if not S:
            f.append(0.0)
            continue
        H = G.subgraph(S)
        f.append(float(H.number_of_edges() - H.number_of_nodes()
                       + nx.number_connected_components(H)))
    return f


# ---------------------------------------------------------------- ESPH block (144)
def diagram_stats(pairs):
    """count, sum/mean/std/max lifetime, mean birth, mean death, persistence entropy."""
    if len(pairs) == 0:
        return [0.0] * 8
    b = np.array([p[0] for p in pairs], dtype=float)
    dd = np.array([p[1] for p in pairs], dtype=float)
    keep = np.isfinite(dd)
    b, dd = b[keep], dd[keep]
    if len(b) == 0:
        return [0.0] * 8
    L = dd - b
    tot = L.sum()
    if tot > 1e-12:
        p = L / tot
        p = p[p > 0]
        E = float(-np.sum(p * np.log(p)))
    else:
        E = 0.0
    return [float(len(L)), float(L.sum()), float(L.mean()), float(L.std()),
            float(L.max()), float(b.mean()), float(dd.mean()), E]


def esph_block(d):
    import gudhi
    sym, X = d['sym'], d['X']
    sels = [
        [i for i, s in enumerate(sym) if s == 'C'],
        [i for i, s in enumerate(sym) if s in ('N', 'O')],
        [i for i, s in enumerate(sym) if s in ('C', 'N', 'O')],
        [i for i, s in enumerate(sym) if s in HALOGENS],
        [i for i, s in enumerate(sym) if s in POLAR],
        list(range(len(sym))),
    ]
    f = []
    for S in sels:
        if len(S) < 1:
            f += [0.0] * 24
            continue
        Y = X[S]
        rc = gudhi.RipsComplex(points=Y, max_edge_length=12.0)
        st = rc.create_simplex_tree(max_dimension=3)
        st.compute_persistence(homology_coeff_field=2, persistence_dim_max=False)
        for k in (0, 1, 2):
            f += diagram_stats(st.persistence_intervals_in_dimension(k))
    return f


# ---------------------------------------------------------------- ADM block (47)
def adm_block(d):
    X, Z, n = d['X'], d['Z'].astype(float), d['n']
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(n, 1)
    off = D[iu]

    f = [float(n)] + moments(off) + order_stats(off)

    # radial profile: 15 bins of 1 A on [0,15) plus a final overflow bin [15, inf)
    h = np.zeros(16)
    if len(off):
        b = np.clip(np.floor(off).astype(int), 0, 15)
        for v in b:
            h[v] += 1
        h /= h.sum()
    f += list(h)

    lam = np.linalg.eigvalsh(D)
    f += pad_spectrum(lam, 6, 4)
    lam_s = np.sort(lam)[::-1]
    f += [float(np.linalg.norm(D)), float(np.max(np.abs(lam))),
          float(lam_s[0] - lam_s[1]) if n > 1 else 0.0,
          float(np.mean(np.abs(lam)))]

    # gyration tensor
    xc = X - X.mean(0)
    T = (xc.T @ xc) / n
    g = np.sort(np.linalg.eigvalsh(T))[::-1]
    Rg2 = float(g.sum())
    Rg = float(np.sqrt(max(Rg2, 0.0)))
    asph = float(g[0] - 0.5 * (g[1] + g[2]))
    acyl = float(g[1] - g[2])
    kappa2 = float((asph ** 2 + 0.75 * acyl ** 2) / (Rg2 ** 2)) if Rg2 > 1e-12 else 0.0
    f += [Rg, float(g[0]), float(g[1]), float(g[2]), asph, acyl, kappa2]

    return f


# ---------------------------------------------------------------- AM block (42)
def am_block(d):
    X, n = d['X'], d['n']
    G = graph_of(d)

    # bond angles at each central atom j, over all distinct pairs of its neighbours
    ang, acc = [], {}
    for j in G.nodes():
        nb = list(G.neighbors(j))
        for i, k in combinations(nb, 2):
            v1, v2 = X[i] - X[j], X[k] - X[j]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            t = float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
            ang.append(t)
            acc.setdefault((min(i, k), max(i, k)), []).append(t)
    ang = np.array(ang, dtype=float)

    # the angle matrix itself
    Theta = np.zeros((n, n))
    for (i, k), vals in acc.items():
        Theta[i, k] = Theta[k, i] = float(np.mean(vals))

    f = [float(len(ang))] + moments(ang) + order_stats(ang)

    h = np.zeros(12)
    if len(ang):
        b = np.clip((ang / np.pi * 12).astype(int), 0, 11)
        for v in b:
            h[v] += 1
        h /= h.sum()
    f += list(h)

    if len(ang):
        deg = np.degrees(ang)
        f += [float(np.mean(np.abs(deg - 109.5) <= 7.5)),
              float(np.mean(np.abs(deg - 120.0) <= 7.5)),
              float(np.mean(deg > 165.0))]
    else:
        f += [0.0] * 3

    lam = np.linalg.eigvalsh(Theta)
    f += pad_spectrum(lam, 5, 3)
    lam_s = np.sort(lam)[::-1]
    f += [float(np.linalg.norm(Theta)), float(np.max(np.abs(lam))),
          float(lam_s[0] - lam_s[1]) if n > 1 else 0.0,
          float(np.mean(np.abs(lam)))]

    # proper torsions about the bonds of G
    tors = []
    for (i, k) in G.edges():
        for j in G.neighbors(i):
            if j == k:
                continue
            for l in G.neighbors(k):
                if l == i or l == j:
                    continue
                b1, b2, b3 = X[i] - X[j], X[k] - X[i], X[l] - X[k]
                n2 = np.linalg.norm(b2)
                if n2 < 1e-9:
                    continue
                c1, c2 = np.cross(b1, b2), np.cross(b2, b3)
                y = np.dot(np.cross(c1, c2), b2 / n2)
                x = np.dot(c1, c2)
                if abs(x) < 1e-12 and abs(y) < 1e-12:
                    continue
                tors.append(np.arctan2(y, x))
    tors = np.array(tors, dtype=float)
    if len(tors):
        ct = np.abs(np.cos(tors))
        deg = np.degrees(np.abs(tors))
        f += [float(len(tors)), float(ct.mean()), float(ct.std()),
              float(np.mean(deg <= 30.0)), float(np.mean(deg >= 150.0))]
    else:
        f += [0.0] * 5
    return f


BLOCKS = [('FG', fg_block, 28), ('ESPH', esph_block, 144),
          ('ADM', adm_block, 47), ('AM', am_block, 42)]


def featurize(d):
    out = {}
    for name, fn, dim in BLOCKS:
        v = np.array(fn(d), dtype=np.float64)
        assert len(v) == dim, f'{name}: {len(v)} != {dim}'
        out[name] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def main():
    shard = sys.argv[1]
    with open(f'conf_{shard}.pkl', 'rb') as fh:
        mols = pickle.load(fh)
    cols = {k: [] for k, _, _ in BLOCKS}
    y, idx = [], []
    for c, d in enumerate(mols):
        try:
            fe = featurize(d)
        except Exception as e:
            print('skip', d['smiles'], e, flush=True)
            continue
        for k in cols:
            cols[k].append(fe[k])
        y.append(d['bp'])
        idx.append(d['idx'])
        if c % 200 == 0:
            print(f'{shard}: {c}/{len(mols)}', flush=True)
    np.savez_compressed(f'feat_{shard}.npz',
                        y=np.array(y), idx=np.array(idx),
                        **{k: np.array(v) for k, v in cols.items()})
    print(f'{shard}: done {len(y)}', flush=True)


if __name__ == '__main__':
    main()
