import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
import numpy as np
import pandas as pd


def parse_mesh_xml(mesh_xml_path: str):
    tree = ET.parse(mesh_xml_path)
    root = tree.getroot()
    ns = {"m": root.tag.split('}')[0].strip('{')} if root.tag.startswith("{") else {}
    mesh_id_to_trees: Dict[str, Set[str]] = defaultdict(set)
    tree_to_mesh_id: Dict[str, str] = {}
    for dr in root.findall(".//m:DescriptorRecord" if ns else ".//DescriptorRecord", ns):
        ui_el = dr.find("m:DescriptorUI" if ns else "DescriptorUI", ns)
        if ui_el is None:
            continue
        mesh_id = ui_el.text.strip()
        tlist = dr.find("m:TreeNumberList" if ns else "TreeNumberList", ns)
        if tlist is None:
            continue
        for tnum_el in tlist.findall("m:TreeNumber" if ns else "TreeNumber", ns):
            tnum = tnum_el.text.strip()
            if tnum:
                mesh_id_to_trees[mesh_id].add(tnum)
                tree_to_mesh_id[tnum] = mesh_id
    return mesh_id_to_trees, tree_to_mesh_id


def build_parent_child_edges(mesh_id_to_trees: Dict[str, Set[str]],
                             tree_to_mesh_id: Dict[str, str]) -> Dict[str, Set[str]]:
    children: Dict[str, Set[str]] = defaultdict(set)
    for mesh_id, tnums in mesh_id_to_trees.items():
        for tn in tnums:
            if '.' in tn:
                parent_tn = tn.rsplit('.', 1)[0]
            else:
                parent_tn = None

            if parent_tn and parent_tn in tree_to_mesh_id:
                parent_mesh = tree_to_mesh_id[parent_tn]
                if parent_mesh != mesh_id:
                    children[parent_mesh].add(mesh_id)

            children.setdefault(mesh_id, set())

    for p, chs in list(children.items()):
        children.setdefault(p, chs)

    return children


class DiseaseDAG:
    def __init__(self, children: Dict[str, Set[str]]):
        self.children = {k: set(v) for k, v in children.items()}
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        for p, chs in self.children.items():
            for c in chs:
                self.parents[c].add(p)
        all_nodes = set(self.children.keys()) | set(self.parents.keys())
        for n in all_nodes:
            self.children.setdefault(n, set())
            self.parents.setdefault(n, set())
        self._anc_cache: Dict[str, Set[str]] = {}
        self._contrib_cache: Dict[Tuple[str, float], Dict[str, float]] = {}
        self._dv_cache: Dict[Tuple[str, float], float] = {}

    def ancestors(self, node: str) -> Set[str]:
        if node in self._anc_cache:
            return self._anc_cache[node]
        vis = set()
        stack = list(self.parents[node])
        while stack:
            u = stack.pop()
            if u not in vis:
                vis.add(u)
                stack.extend(self.parents[u])
        self._anc_cache[node] = vis
        return vis

    def N(self, d: str) -> Set[str]:
        return {d} | self.ancestors(d)

    def _contrib_bottom_up(self, d: str, delta: float) -> Dict[str, float]:
        key = (d, delta)
        if key in self._contrib_cache:
            return self._contrib_cache[key]

        Nd = self.N(d)
        sub_children = {n: (self.children[n] & Nd) for n in Nd}
        sub_parents = {n: (self.parents[n] & Nd) for n in Nd}
        outdeg = {n: len(sub_children[n]) for n in Nd}

        C = {n: 0.0 for n in Nd}
        C[d] = 1.0

        processed = set()
        for _ in range(len(Nd) + 5):
            leaves = [n for n in Nd if outdeg[n] == 0 and n not in processed]
            if not leaves:
                break
            for n in leaves:
                processed.add(n)
                for p in sub_parents[n]:
                    C[p] = max(C[p], delta * C[n])
                    outdeg[p] -= 1

        self._contrib_cache[key] = C
        return C

    def DV(self, d: str, delta: float) -> float:
        key = (d, delta)
        if key in self._dv_cache:
            return self._dv_cache[key]
        C = self._contrib_bottom_up(d, delta)
        val = float(sum(C.values()))
        self._dv_cache[key] = val
        return val

    def sim(self, di: str, dj: str, delta: float) -> float:
        Ndi, Ndj = self.N(di), self.N(dj)
        inter = Ndi & Ndj
        Ci = self._contrib_bottom_up(di, delta)
        Cj = self._contrib_bottom_up(dj, delta)
        num = sum((Ci[n] + Cj[n]) for n in inter)
        den = self.DV(di, delta) + self.DV(dj, delta)
        return float(num / den) if den > 0 else 0.0

    def similarity_matrix(self, diseases: List[str],
                          delta: float = 0.5,
                          topk: Optional[int] = 15,
                          binarize: bool = True,
                          keep_diag: bool = False) -> np.ndarray:
        n = len(diseases)
        S = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    S[i, j] = 1.0
                else:
                    s = self.sim(diseases[i], diseases[j], delta)
                    S[i, j] = S[j, i] = s

        if not keep_diag:
            np.fill_diagonal(S, 0.0)

        if topk is not None and topk > 0:
            S_new = np.zeros_like(S)
            for i in range(n):
                mask = np.arange(n) != i
                vals = S[i, mask]
                cols = np.arange(n)[mask]
                k = min(topk, vals.size)
                if k > 0 and vals.size > 0:
                    top_idx = cols[vals.argsort()[::-1][:k]]
                    if binarize:
                        S_new[i, top_idx] = 1.0
                    else:
                        S_new[i, top_idx] = S[i, top_idx]
            S = np.maximum(S_new, S_new.T)
        return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_xml", type=str, required=True)
    ap.add_argument("--disease_file", type=str, required=True)
    ap.add_argument("--sep", type=str, default=",")
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--binarize", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True)
    ap.add_argument("--keep_diag", action="store_true")
    ap.add_argument("--out_matrix", type=str, required=True)
    ap.add_argument("--out_edges", type=str, required=True)
    args = ap.parse_args()

    mesh_id_to_trees, tree_to_mesh_id = parse_mesh_xml(args.mesh_xml)
    if args.disease_file.endswith(".xlsx") or args.disease_file.endswith(".xls"):
        df = pd.read_excel(args.disease_file)
    else:
        df = pd.read_csv(args.disease_file, sep=args.sep)

    assert {"Disease", "MESH ID"} <= set(df.columns), "Disease, MESH ID"
    diseases = df["MESH ID"].astype(str).tolist()
    names = df["Disease"].astype(str).tolist()
    children = build_parent_child_edges(mesh_id_to_trees, tree_to_mesh_id)
    dag = DiseaseDAG(children)
    S = dag.similarity_matrix(diseases, delta=args.delta, topk=args.topk,
                              binarize=args.binarize, keep_diag=args.keep_diag)
    idx = [f"{n} ({m})" for n, m in zip(names, diseases)]
    pd.DataFrame(S, index=idx, columns=idx).to_csv(args.out_matrix, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"[save] matrix: {args.out_matrix} shape={S.shape}")
    edges = []
    n = len(diseases)
    for i in range(n):
        for j in range(i+1, n):
            if S[i, j] > 0:
                edges.append((diseases[i], diseases[j], S[i, j]))
    pd.DataFrame(edges, columns=["source", "target", "score"]).to_csv(args.out_edges, index=False)
    print(f"[save] edges: {args.out_edges} count={len(edges)}")


if __name__ == "__main__":
    main()
