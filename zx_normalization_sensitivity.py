#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZX Normalization Sensitivity
============================

Experimental framework for studying representation-sensitive dependency
abstractions in ZX-normalized Clifford+T circuits.

The pipeline evaluates the map

    C  ->  N(C)  ->  G_T(N(C))  ->  chi(G_T(N(C)))

where C is an input circuit, N is a semantics-preserving normalization or
extraction strategy, G_T is the induced T-dependency graph, and chi denotes
the graph chromatic number.

The script is intended for reproducibility of the experiments reported in

    Semantic Equivalence and Dependency-Graph Sensitivity in
    ZX-Normalized Clifford+T Circuits

Main steps
----------
1. Generate structured and randomized Clifford+T benchmark circuits.
2. Apply several ZX/Qiskit/tket normalization and extraction strategies.
3. Retain only variants passing the equivalence checks used in the analysis.
4. Construct T-dependency graphs from the normalized representations.
5. Compute graph-optimal T-layer values through SAT-based graph coloring.
6. Record graph diversity, chromatic diversity, replay gaps, and family-level
   sensitivity summaries.
7. Export CSV, JSON, LaTeX, GraphML, and figure artifacts.

Dependencies
------------
Required:
    qiskit, numpy, pandas, networkx, matplotlib, python-sat

Recommended:
    pyzx, mqt.qcec, pytket, pynauty, scipy

Typical runs
------------
Quick sanity check:
    python zx_normalization_sensitivity.py --outdir artifacts_normsens --quick

Medium experiment:
    python zx_normalization_sensitivity.py --outdir artifacts_normsens --medium --pairwise-equivalence

Larger experiment:
    python zx_normalization_sensitivity.py --outdir artifacts_normsens --large --pairwise-equivalence --random-seeds 0 1 2 3 4

External OpenQASM 2 benchmark directory:
    python zx_normalization_sensitivity.py --outdir artifacts_normsens --medium --qasm-dir path/to/qasm

Notes
-----
Failed extraction or failed equivalence checks are quarantined and are not used
as evidence for normalization sensitivity. Sensitivity claims are based only on
variants that pass the equivalence guard.
"""

from __future__ import annotations

import argparse
import signal
import csv
import dataclasses
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
import traceback
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*implicit qubit permutations.*")
import zipfile

# Keep long Spyder runs readable. These warnings do not affect the
# normalization-sensitivity analysis and are recorded only as environment noise.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"qiskit\..*")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module=r"qiskit\..*")
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx



# ---------------------------------------------------------------------------
# GraphML export helpers
# ---------------------------------------------------------------------------
# NetworkX's GraphML writer accepts only primitive scalar attributes.  The
# dependency graphs used in this script deliberately store rich provenance
# values such as tuples/lists/dicts (e.g. qubit sets or blocker descriptions).
# Those are valid Python/JSON artifacts but invalid GraphML attributes.  Before
# writing .graphml files, sanitize every node/edge attribute without changing
# the graph used for analysis.

def _graphml_safe_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return str(value)

def graphml_safe_copy(G: nx.Graph) -> nx.Graph:
    H = G.__class__()
    H.graph.update({str(k): _graphml_safe_scalar(v) for k, v in G.graph.items()})
    for node, data in G.nodes(data=True):
        H.add_node(str(node), **{str(k): _graphml_safe_scalar(v) for k, v in data.items()})
    for u, v, data in G.edges(data=True):
        H.add_edge(str(u), str(v), **{str(k): _graphml_safe_scalar(val) for k, val in data.items()})
    return H

def write_graphml_safe(G: nx.Graph, path: Path) -> None:
    nx.write_graphml(graphml_safe_copy(G), path)


try:
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit.*")
    warnings.filterwarnings("ignore", message=".*EfficientSU2.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*RealAmplitudes.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*implicit qubit permutations.*")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import QuantumRegister
    from qiskit.circuit.library import EfficientSU2, QFT, TwoLocal, RealAmplitudes, MCXGate, RZGate, RXGate, RYGate
    from qiskit.quantum_info import Statevector, Operator
    try:
        from qiskit.qasm2 import dumps as qasm2_dumps, loads as qasm2_loads
    except Exception:
        qasm2_dumps = None
        qasm2_loads = None
    # Run policy: do not use QASM3 anywhere in the
    # normalization pipeline. QASM3 optional importers created environment
    # noise in earlier runs. External .qasm3 files are skipped explicitly.
    qasm3 = None
    HAVE_QISKIT = True
except Exception as e:
    HAVE_QISKIT = False
    QISKIT_IMPORT_ERROR = str(e)

try:
    import pyzx as zx
    HAVE_PYZX = True
except Exception:
    HAVE_PYZX = False

try:
    from pysat.formula import CNF
    from pysat.solvers import Minisat22, Glucose3
    HAVE_PYSAT = True
except Exception:
    HAVE_PYSAT = False

try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import pynauty
    HAVE_PYNAUTY = True
except Exception:
    HAVE_PYNAUTY = False

try:
    from mqt import qcec as mqt_qcec
    HAVE_QCEC = True
except Exception:
    HAVE_QCEC = False

# tket API changed across versions. Import converters and the passes module
# separately, then choose an available pass dynamically. Earlier versions of
# this script imported PeepholeOptimise directly, which fails on newer pytket
# versions even when pytket itself is installed correctly.
try:
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    import pytket.passes as _tket_passes
    HAVE_TKET = True
    TKET_IMPORT_ERROR = None
    TKET_AVAILABLE_PASSES = [name for name in dir(_tket_passes) if not name.startswith("_")]
except Exception as _tket_e:
    HAVE_TKET = False
    TKET_IMPORT_ERROR = repr(_tket_e)
    TKET_AVAILABLE_PASSES = []

def _select_tket_pass():
    """Return (pass_name, pass_instance) for the active pytket version.

    Preference order keeps the intended role: an external compiler-level
    normalization/optimization baseline. If no optimization pass is available,
    return (None, None) and the tket strategy performs a conversion-only
    normalization rather than failing.
    """
    if not HAVE_TKET:
        return None, None
    candidates = [
        "FullPeepholeOptimise",
        "PeepholeOptimise",
        "CliffordSimp",
        "SynthesiseTket",
        "SynthesiseTK",
        "RebaseTket",
    ]
    for cname in candidates:
        cls = getattr(_tket_passes, cname, None)
        if cls is None:
            continue
        try:
            return cname, cls()
        except Exception:
            continue
    return None, None

ANGLE_TOL_DEFAULT = 1e-7
T_NAMES = {"t", "tdg"}
Z_PHASE_NAMES = {"z", "s", "sdg", "t", "tdg", "rz", "p", "u1"}



class TimeoutExpired(Exception):
    pass

class time_limit:
    """Unix-only soft timeout used to prevent a single normalization/equivalence
    branch from blocking a long publication run. It does not alter the scientific
    pipeline: timed-out branches are quarantined with failure_category=timeout.
    """
    def __init__(self, seconds: int | float | None):
        try:
            self.seconds = int(seconds) if seconds is not None else 0
        except Exception:
            self.seconds = 0
        self.old_handler = None

    def _handle(self, signum, frame):
        raise TimeoutExpired(f"timeout after {self.seconds}s")

    def __enter__(self):
        if self.seconds and self.seconds > 0 and hasattr(signal, "SIGALRM"):
            self.old_handler = signal.signal(signal.SIGALRM, self._handle)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds and self.seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if self.old_handler is not None:
                signal.signal(signal.SIGALRM, self.old_handler)
        return False

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(out)[:180]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def json_dump(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def text_write(path: str | Path, txt: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def append_log(path: str | Path, msg: str) -> None:
    ensure_dir(Path(path).parent)
    line = f"[{now()}] {msg}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    print(msg)


def ffloat(x: Any, nd: int = 6) -> Any:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return round(float(x), nd)
    except Exception:
        return x

# ---------------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------------

def qaoa_ring(n: int, p: int, gamma: float = math.pi / 8, beta: float = math.pi / 8) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"qaoa_ring_n{n}_p{p}")
    for i in range(n): qc.h(i)
    for _ in range(p):
        for i in range(n):
            j = (i + 1) % n
            qc.cx(i, j); qc.rz(2 * gamma, j); qc.cx(i, j)
        for i in range(n): qc.rx(2 * beta, i)
    return qc


def qaoa_grid_2d(rows: int, cols: int, p: int, gamma: float = math.pi / 8, beta: float = math.pi / 8) -> QuantumCircuit:
    n = rows * cols
    qc = QuantumCircuit(n, name=f"qaoa_grid_{rows}x{cols}_p{p}")
    def idx(r, c): return r * cols + c
    for i in range(n): qc.h(i)
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols: edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows: edges.append((idx(r, c), idx(r + 1, c)))
    for _ in range(p):
        for i, j in edges:
            qc.cx(i, j); qc.rz(2 * gamma, j); qc.cx(i, j)
        for i in range(n): qc.rx(2 * beta, i)
    return qc


def qaoa_complete(n: int, p: int, gamma: float = math.pi / 8, beta: float = math.pi / 8) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"qaoa_complete_n{n}_p{p}")
    for i in range(n): qc.h(i)
    for _ in range(p):
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j); qc.rz(2 * gamma, j); qc.cx(i, j)
        for i in range(n): qc.rx(2 * beta, i)
    return qc


def random_ct(n: int, depth: int, seed: int) -> QuantumCircuit:
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"random_ct_n{n}_d{depth}_s{seed}")
    for layer in range(depth):
        for q in range(n):
            r = rnd.random()
            if r < 0.25: qc.h(q)
            elif r < 0.45: qc.s(q)
            elif r < 0.70: qc.t(q)
            elif r < 0.85: qc.tdg(q)
            else: qc.rz(rnd.choice([math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4]), q)
        perm = list(range(n)); rnd.shuffle(perm)
        for a, b in zip(perm[::2], perm[1::2]): qc.cx(a, b)
    return qc


def random_interaction_ct(n: int, depth: int, seed: int, p_edge: float = 0.25) -> QuantumCircuit:
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"random_interaction_ct_n{n}_d{depth}_s{seed}")
    for layer in range(depth):
        for q in range(n):
            if rnd.random() < 0.65:
                qc.rz(rnd.choice([math.pi/4, -math.pi/4, math.pi/2, -math.pi/2]), q)
            elif rnd.random() < 0.5:
                qc.h(q)
        for i in range(n):
            for j in range(i+1, n):
                if rnd.random() < p_edge:
                    qc.cx(i, j)
    return qc


def brickwork_ct(n: int, layers: int, seed: int = 0) -> QuantumCircuit:
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"brickwork_ct_n{n}_l{layers}_s{seed}")
    for l in range(layers):
        start = l % 2
        for i in range(start, n - 1, 2): qc.cx(i, i + 1)
        for q in range(n):
            qc.rz(rnd.choice([math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4]), q)
            if rnd.random() < 0.35: qc.h(q)
    return qc


def dense_phase(n: int, rounds: int = 1) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"dense_phase_n{n}_r{rounds}")
    for _ in range(rounds):
        for i in range(n): qc.h(i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j); qc.t(j); qc.cx(i, j)
        for i in range(n): qc.h(i)
    return qc


def qft_phase_like(n: int) -> QuantumCircuit:
    # Clifford+T-like coarse discretized QFT-inspired phase ladder.
    qc = QuantumCircuit(n, name=f"qft_phase_like_n{n}")
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            # use pi/4 for T-like phases to keep the analysis in Clifford+T-like set
            qc.cx(j, i); qc.rz(math.pi / 4, i); qc.cx(j, i)
    return qc


def vqe_su2_bound(n: int, reps: int, seed: int = 0) -> QuantumCircuit:
    qc = EfficientSU2(num_qubits=n, reps=reps, entanglement="linear")
    # Bind to Clifford+T angles to make T-like detection meaningful.
    params = sorted(qc.parameters, key=lambda p: p.name)
    rnd = random.Random(seed)
    choices = [0, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2]
    mapping = {p: rnd.choice(choices) for p in params}
    out = qc.assign_parameters(mapping, inplace=False)
    out.name = f"vqe_su2_n{n}_r{reps}_s{seed}"
    return out


def real_amplitudes_bound(n: int, reps: int, seed: int = 0) -> QuantumCircuit:
    qc = RealAmplitudes(num_qubits=n, reps=reps, entanglement="linear")
    params = sorted(qc.parameters, key=lambda p: p.name)
    rnd = random.Random(seed)
    choices = [0, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2]
    out = qc.assign_parameters({p: rnd.choice(choices) for p in params}, inplace=False)
    out.name = f"realamp_n{n}_r{reps}_s{seed}"
    return out


def _get_or_make_ancilla_register(qc: QuantumCircuit, need: int, base: str = "anc") -> QuantumRegister:
    """Return an existing ancilla register if it is large enough; otherwise create a uniquely named one.

    Qiskit does not allow two QuantumRegisters with the same name. The earlier v5 implementation
    always tried to add QuantumRegister(..., "anc"), which fails for Grover-like circuits because
    the oracle and diffuser both call _mcx on the same circuit.
    """
    need = max(1, int(need))
    for qreg in qc.qregs:
        if qreg.name.startswith(base) and len(qreg) >= need:
            return qreg
    existing = {qreg.name for qreg in qc.qregs}
    name = base
    idx = 0
    while name in existing:
        idx += 1
        name = f"{base}_{idx}"
    anc = QuantumRegister(need, name)
    qc.add_register(anc)
    return anc


def _mcx(qc: QuantumCircuit, controls: List[int], target: int) -> None:
    if len(controls) <= 4:
        qc.mcx(controls, target)
    else:
        need = max(1, len(controls) - 2)
        anc = _get_or_make_ancilla_register(qc, need)
        qc.mcx(controls, target, ancilla_qubits=list(anc)[:need])


def grover_like(n: int, iterations: int = 1) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"grover_like_n{n}_it{iterations}")
    for i in range(n): qc.h(i)
    for _ in range(iterations):
        qc.h(n-1); _mcx(qc, list(range(n-1)), n-1); qc.h(n-1)
        for i in range(n): qc.h(i); qc.x(i)
        qc.h(n-1); _mcx(qc, list(range(n-1)), n-1); qc.h(n-1)
        for i in range(n): qc.x(i); qc.h(i)
    return qc


def tof_chain(n: int) -> QuantumCircuit:
    # Toffoli ladder, structured arithmetic-like non-Clifford structure.
    qc = QuantumCircuit(n, name=f"tof_chain_n{n}")
    for i in range(max(0, n - 2)):
        qc.ccx(i, i+1, i+2)
    for i in reversed(range(max(0, n - 2))):
        qc.ccx(i, i+1, i+2)
    return qc


def phase_polynomial_stress(n: int, terms: int, seed: int) -> QuantumCircuit:
    # Parity-phase circuit: compute random parity onto pivot, apply T-like phase, uncompute.
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"phase_poly_n{n}_t{terms}_s{seed}")
    for _ in range(terms):
        subset = sorted(rnd.sample(range(n), rnd.randint(2, min(n, max(2, n//2 + 1)))))
        pivot = subset[-1]
        for q in subset[:-1]: qc.cx(q, pivot)
        qc.rz(rnd.choice([math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4]), pivot)
        for q in reversed(subset[:-1]): qc.cx(q, pivot)
    return qc


@dataclass
class BenchmarkSpec:
    family: str
    circuit_id: str
    n: int  # physical qubit count in the constructed circuit
    size_param: int  # logical/problem-size parameter when applicable
    seed: int
    circuit: Any

    @property
    def physical_qubits(self) -> int:
        try:
            return int(getattr(self.circuit, "num_qubits", self.n))
        except Exception:
            return int(self.n)

    @property
    def logical_size(self) -> int:
        try:
            return int(self.size_param)
        except Exception:
            return int(self.n)



def ripple_carry_toffoli_stress(n: int) -> QuantumCircuit:
    """Small arithmetic-style benchmark: a reversible carry chain rich in CCX/T after Clifford+T decomposition."""
    # Registers: a[0..n-1], b[0..n-1], carry[0..n]
    qc = QuantumCircuit(2*n + n + 1, name=f"ripple_carry_toffoli_n{n}")
    a = list(range(n)); b = list(range(n, 2*n)); c = list(range(2*n, 3*n+1))
    for i in range(n):
        qc.cx(a[i], b[i])
        qc.ccx(a[i], b[i], c[i+1])
        qc.cx(c[i], b[i])
        if i + 1 < n:
            qc.ccx(c[i+1], b[i], c[i+2])
    # Partial uncompute to avoid a purely monotone chain while keeping arithmetic structure.
    for i in reversed(range(max(0, n-1))):
        qc.ccx(c[i+1], b[i], c[i+2])
    return qc


def controlled_phase_ladder(n: int, depth: int) -> QuantumCircuit:
    """QFT/arithmetic-inspired controlled phase ladder with interleaved basis changes."""
    qc = QuantumCircuit(n, name=f"controlled_phase_ladder_n{n}_d{depth}")
    for q in range(n): qc.h(q)
    for d in range(depth):
        for i in range(n-1):
            j = i + 1
            qc.cx(i, j); qc.rz(((-1)**(i+d)) * math.pi/4, j); qc.cx(i, j)
        if d % 2 == 0:
            for q in range(0, n, 2): qc.h(q)
        else:
            for q in range(1, n, 2): qc.h(q)
    return qc


def random_cnot_t_layered(n: int, layers: int, seed: int) -> QuantumCircuit:
    """Random Clifford+T benchmark closer to synthesis/optimization stress tests."""
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"random_cnot_t_n{n}_l{layers}_s{seed}")
    for l in range(layers):
        qs = list(range(n)); rnd.shuffle(qs)
        for q in qs:
            if rnd.random() < 0.55:
                qc.t(q) if rnd.random() < 0.7 else qc.tdg(q)
            elif rnd.random() < 0.5:
                qc.h(q)
            else:
                qc.s(q)
        pairs = list(range(n)); rnd.shuffle(pairs)
        for a,b in zip(pairs[::2], pairs[1::2]):
            qc.cx(a,b)
        if rnd.random() < 0.35 and n >= 3:
            a,b,c = rnd.sample(range(n), 3)
            qc.ccx(a,b,c)
    return qc



def qaoa_from_edges(n: int, edges: Sequence[Tuple[int, int]], p: int, name: str,
                    gamma: float = math.pi / 8, beta: float = math.pi / 8) -> QuantumCircuit:
    """Generic QAOA-style phase separator for graph families used in modern benchmarks."""
    qc = QuantumCircuit(n, name=f"{name}_n{n}_p{p}")
    for i in range(n): qc.h(i)
    clean_edges = [(int(a), int(b)) for a, b in edges if int(a) != int(b)]
    for _ in range(p):
        for i, j in clean_edges:
            qc.cx(i, j); qc.rz(2 * gamma, j); qc.cx(i, j)
        for i in range(n): qc.rx(2 * beta, i)
    return qc


def qaoa_path(n: int, p: int) -> QuantumCircuit:
    return qaoa_from_edges(n, [(i, i+1) for i in range(n-1)], p, "qaoa_path")


def qaoa_star(n: int, p: int) -> QuantumCircuit:
    return qaoa_from_edges(n, [(0, i) for i in range(1, n)], p, "qaoa_star")


def qaoa_random_regular(n: int, degree: int, p: int, seed: int) -> QuantumCircuit:
    degree = min(degree, n-1)
    if (n * degree) % 2 == 1:
        degree = max(1, degree-1)
    G = nx.random_regular_graph(degree, n, seed=seed) if n > degree and degree > 0 else nx.path_graph(n)
    return qaoa_from_edges(n, list(G.edges()), p, f"qaoa_rr{degree}_s{seed}")


def qaoa_erdos_renyi(n: int, p_edge: float, p: int, seed: int) -> QuantumCircuit:
    G = nx.erdos_renyi_graph(n, p_edge, seed=seed)
    if G.number_of_edges() == 0 and n > 1:
        G.add_edges_from((i, i+1) for i in range(n-1))
    return qaoa_from_edges(n, list(G.edges()), p, f"qaoa_er{str(p_edge).replace('.','p')}_s{seed}")


def qaoa_barabasi_albert(n: int, m: int, p: int, seed: int) -> QuantumCircuit:
    m = max(1, min(m, n-1))
    G = nx.barabasi_albert_graph(n, m, seed=seed) if n > 2 else nx.path_graph(n)
    return qaoa_from_edges(n, list(G.edges()), p, f"qaoa_ba{m}_s{seed}")


def commuting_iqp(n: int, terms: int, seed: int) -> QuantumCircuit:
    """IQP/diagonal commuting benchmark: important negative-control family."""
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"iqp_commuting_n{n}_t{terms}_s{seed}")
    for q in range(n): qc.h(q)
    for _ in range(terms):
        arity = rnd.choice([1, 2, 3]) if n >= 3 else rnd.choice([1, 2])
        subset = sorted(rnd.sample(range(n), min(arity, n)))
        if len(subset) == 1:
            qc.rz(rnd.choice([math.pi/4, -math.pi/4]), subset[0])
        else:
            pivot = subset[-1]
            for q in subset[:-1]: qc.cx(q, pivot)
            qc.rz(rnd.choice([math.pi/4, -math.pi/4]), pivot)
            for q in reversed(subset[:-1]): qc.cx(q, pivot)
    for q in range(n): qc.h(q)
    return qc


def bv_oracle_like(n: int, seed: int) -> QuantumCircuit:
    """Bernstein-Vazirani-like oracle with injected T-like phases to avoid all-Clifford triviality."""
    rnd = random.Random(seed)
    qc = QuantumCircuit(n+1, name=f"bv_phase_like_n{n}_s{seed}")
    out = n
    qc.x(out); qc.h(out)
    for q in range(n): qc.h(q)
    secret = [rnd.randint(0,1) for _ in range(n)]
    if not any(secret): secret[0] = 1
    for q,b in enumerate(secret):
        if b:
            qc.cx(q, out); qc.rz(math.pi/4, q); qc.cx(q, out)
    for q in range(n): qc.h(q)
    return qc


def hardware_efficient_ct(n: int, reps: int, seed: int, entanglement: str = "linear") -> QuantumCircuit:
    """Hardware-efficient ansatz with Clifford+T parameter binding."""
    rnd = random.Random(seed)
    qc = QuantumCircuit(n, name=f"hea_{entanglement}_n{n}_r{reps}_s{seed}")
    choices = [0, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2]
    for r in range(reps):
        for q in range(n):
            qc.rx(rnd.choice(choices), q); qc.rz(rnd.choice(choices), q)
        if entanglement == "circular":
            for q in range(n): qc.cx(q, (q+1)%n)
        elif entanglement == "full":
            for i in range(n):
                for j in range(i+1, n): qc.cx(i, j)
        else:
            for q in range(n-1): qc.cx(q, q+1)
    return qc


def modular_adder_proxy(n: int) -> QuantumCircuit:
    """Arithmetic proxy with Toffoli/CNOT carry motifs; scalable but still generator-only."""
    qc = QuantumCircuit(3*n + 2, name=f"modular_adder_proxy_n{n}")
    a = list(range(n)); b = list(range(n, 2*n)); c = list(range(2*n, 3*n+2))
    for i in range(n):
        qc.ccx(a[i], b[i], c[i+1])
        qc.cx(a[i], b[i])
        qc.ccx(c[i], b[i], c[i+1])
        qc.t(b[i]); qc.tdg(c[i+1])
    for i in reversed(range(n)):
        qc.ccx(c[i], b[i], c[i+1])
        qc.cx(a[i], b[i])
        qc.ccx(a[i], b[i], c[i+1])
    return qc


def multiplier_proxy(n: int) -> QuantumCircuit:
    """Small multiplier-like Toffoli mesh; use cautiously for large n."""
    qc = QuantumCircuit(3*n, name=f"multiplier_proxy_n{n}")
    a = list(range(n)); b = list(range(n, 2*n)); acc = list(range(2*n, 3*n))
    for i in range(n):
        for j in range(n):
            qc.ccx(a[i], b[j], acc[(i+j) % n])
            if (i+j) % 2 == 0: qc.t(acc[(i+j) % n])
    return qc


def size_class_from_counts(n_qubits: int, t_count: Optional[int], ops: Optional[int] = None) -> str:
    tc = int(t_count or 0); op = int(ops or 0)
    if n_qubits <= 8 and tc < 100 and op < 1000: return "tiny"
    if n_qubits <= 16 and tc < 500 and op < 5000: return "small"
    if n_qubits <= 32 and tc < 3000 and op < 50000: return "medium"
    return "large_proxy"


def make_benchmarks(mode: str, seeds: Sequence[int], include_grover: bool = False,
                    include_qaoa_p2_large: bool = False, max_qaoa_p2_n: int = 10,
                    include_random_cnot_t_large: bool = False, max_random_cnot_t_n: int = 10,
                    include_hea_large: bool = False, max_hea_n: int = 12,
                    include_qaoa_rr_large: bool = False, max_qaoa_rr_n: int = 10) -> List[BenchmarkSpec]:
    if not HAVE_QISKIT:
        raise RuntimeError(f"Qiskit not available: {globals().get('QISKIT_IMPORT_ERROR','unknown')}")
    specs: List[BenchmarkSpec] = []
    seen = set()
    def add(fam: str, cid: str, n: int, sp: int, seed: int, qc: QuantumCircuit):
        sid = safe_name(cid)
        key = (fam, sid, seed)
        if key in seen: return
        seen.add(key)
        specs.append(BenchmarkSpec(fam, sid, n, sp, seed, qc))

    if mode == "quick":
        ns_small = [4, 5, 6]
        for n in ns_small:
            add("qaoa_ring", f"qaoa_ring_n{n}_p1", n, 1, 0, qaoa_ring(n, 1))
            add("qaoa_path", f"qaoa_path_n{n}_p1", n, 1, 0, qaoa_path(n, 1))
            add("qaoa_complete", f"qaoa_complete_n{n}_p1", n, 1, 0, qaoa_complete(n, 1))
            add("dense_phase", f"dense_phase_n{n}_r1", n, 1, 0, dense_phase(n, 1))
            add("brickwork_ct", f"brickwork_n{n}_l3", n, 3, 0, brickwork_ct(n, 3, 0))
        for seed in seeds[:2]:
            add("random_ct", f"random_ct_n6_d4_s{seed}", 6, 4, seed, random_ct(6, 4, seed))
            add("phase_poly", f"phase_poly_n6_t8_s{seed}", 6, 8, seed, phase_polynomial_stress(6, 8, seed))
            add("iqp_commuting", f"iqp_commuting_n6_t8_s{seed}", 6, 8, seed, commuting_iqp(6, 8, seed))
            add("random_cnot_t", f"random_cnot_t_n6_l8_s{seed}", 6, 8, seed, random_cnot_t_layered(6, 8, seed))
        return specs

    if mode == "medium":
        ns = [4, 6, 8, 10, 12, 14]
        for n in ns:
            pvals = [1, 2] if n <= max_qaoa_p2_n else [1]
            for pval in pvals:
                add("qaoa_ring", f"qaoa_ring_n{n}_p{pval}", n, pval, 0, qaoa_ring(n, pval))
                add("qaoa_path", f"qaoa_path_n{n}_p{pval}", n, pval, 0, qaoa_path(n, pval))
                add("qaoa_star", f"qaoa_star_n{n}_p{pval}", n, pval, 0, qaoa_star(n, pval))
            if n <= 10: add("qaoa_complete", f"qaoa_complete_n{n}_p1", n, 1, 0, qaoa_complete(n, 1))
            add("dense_phase", f"dense_phase_n{n}_r1", n, 1, 0, dense_phase(n, 1))
            add("qft_phase_like", f"qft_phase_like_n{n}", n, n, 0, qft_phase_like(n))
            add("controlled_phase_ladder", f"controlled_phase_ladder_n{n}_d2", n, 2, 0, controlled_phase_ladder(n, 2))
            add("brickwork_ct", f"brickwork_n{n}_l4", n, 4, 0, brickwork_ct(n, 4, 0))
            if include_hea_large or n <= max_hea_n:
                add("hardware_efficient_linear", f"hea_linear_n{n}_r2", n, 2, 0, hardware_efficient_ct(n, 2, 0, "linear"))
            if n <= 8:
                add("vqe_su2", f"vqe_su2_n{n}_r2", n, 2, 0, vqe_su2_bound(n, 2, 0))
                add("real_amplitudes", f"realamp_n{n}_r2", n, 2, 0, real_amplitudes_bound(n, 2, 0))
            if include_grover and 5 <= n <= 10: add("grover_like", f"grover_like_n{n}_it1", n, 1, 0, grover_like(n, 1))
            if n >= 5: add("tof_chain", f"tof_chain_n{n}", n, n, 0, tof_chain(n))
            if n <= 10: add("modular_adder_proxy", f"modular_adder_proxy_n{n//2}", 3*(n//2)+2, n//2, 0, modular_adder_proxy(max(2,n//2)))
        for seed in seeds:
            for n in [6, 8, 10, 12, 14]:
                add("qaoa_er", f"qaoa_er_n{n}_p1_s{seed}", n, 1, seed, qaoa_erdos_renyi(n, 0.25, 1, seed))
                if include_qaoa_rr_large or n <= max_qaoa_rr_n:
                    add("qaoa_rr", f"qaoa_rr_n{n}_d3_p1_s{seed}", n, 1, seed, qaoa_random_regular(n, 3, 1, seed))
                add("qaoa_ba", f"qaoa_ba_n{n}_m2_p1_s{seed}", n, 1, seed, qaoa_barabasi_albert(n, 2, 1, seed))
                add("random_ct", f"random_ct_n{n}_d6_s{seed}", n, 6, seed, random_ct(n, 6, seed))
                add("random_interaction_ct", f"random_interaction_ct_n{n}_d4_s{seed}", n, 4, seed, random_interaction_ct(n, 4, seed, p_edge=0.20))
                add("phase_poly", f"phase_poly_n{n}_t{2*n}_s{seed}", n, 2*n, seed, phase_polynomial_stress(n, 2*n, seed))
                add("iqp_commuting", f"iqp_commuting_n{n}_t{2*n}_s{seed}", n, 2*n, seed, commuting_iqp(n, 2*n, seed))
                if include_random_cnot_t_large or n <= max_random_cnot_t_n:
                    add("random_cnot_t", f"random_cnot_t_n{n}_l{2*n}_s{seed}", n, 2*n, seed, random_cnot_t_layered(n, 2*n, seed))
                add("bv_phase_like", f"bv_phase_like_n{n}_s{seed}", n+1, n, seed, bv_oracle_like(n, seed))
        return specs

    # larger generated benchmark suite
    ns = [6, 8, 10, 12, 14, 16, 18, 20, 24]
    for n in ns:
        add("qaoa_ring", f"qaoa_ring_n{n}_p1", n, 1, 0, qaoa_ring(n, 1))
        if include_qaoa_p2_large or n <= max_qaoa_p2_n:
            add("qaoa_ring", f"qaoa_ring_n{n}_p2", n, 2, 0, qaoa_ring(n, 2))
        add("qaoa_path", f"qaoa_path_n{n}_p1", n, 1, 0, qaoa_path(n, 1))
        add("qaoa_star", f"qaoa_star_n{n}_p1", n, 1, 0, qaoa_star(n, 1))
        if n <= 14: add("qaoa_complete", f"qaoa_complete_n{n}_p1", n, 1, 0, qaoa_complete(n, 1))
        if n in [9, 16]:
            r = int(math.sqrt(n)); add("qaoa_grid", f"qaoa_grid_{r}x{r}_p1", n, 1, 0, qaoa_grid_2d(r, r, 1))
        add("dense_phase", f"dense_phase_n{n}_r1", n, 1, 0, dense_phase(n, 1))
        add("qft_phase_like", f"qft_phase_like_n{n}", n, n, 0, qft_phase_like(n))
        add("controlled_phase_ladder", f"controlled_phase_ladder_n{n}_d3", n, 3, 0, controlled_phase_ladder(n, 3))
        add("brickwork_ct", f"brickwork_n{n}_l6", n, 6, 0, brickwork_ct(n, 6, 0))
        if include_hea_large or n <= max_hea_n:
            add("hardware_efficient_linear", f"hea_linear_n{n}_r3", n, 3, 0, hardware_efficient_ct(n, 3, 0, "linear"))
        if n <= 12 and (include_hea_large or n <= max_hea_n): add("hardware_efficient_circular", f"hea_circular_n{n}_r2", n, 2, 0, hardware_efficient_ct(n, 2, 0, "circular"))
        if include_grover and n <= 10: add("grover_like", f"grover_like_n{n}_it1", n, 1, 0, grover_like(n, 1))
        add("tof_chain", f"tof_chain_n{n}", n, n, 0, tof_chain(n))
        if n <= 16: add("modular_adder_proxy", f"modular_adder_proxy_n{max(2,n//2)}", 3*max(2,n//2)+2, max(2,n//2), 0, modular_adder_proxy(max(2,n//2)))
        if n <= 10: add("multiplier_proxy", f"multiplier_proxy_n{max(2,n//3)}", 3*max(2,n//3), max(2,n//3), 0, multiplier_proxy(max(2,n//3)))
    for seed in seeds:
        for n in [8, 10, 12, 14, 16, 18, 20, 24]:
            add("qaoa_er", f"qaoa_er_n{n}_p1_s{seed}", n, 1, seed, qaoa_erdos_renyi(n, 0.18, 1, seed))
            if include_qaoa_rr_large or n <= max_qaoa_rr_n:
                add("qaoa_rr", f"qaoa_rr_n{n}_d3_p1_s{seed}", n, 1, seed, qaoa_random_regular(n, 3, 1, seed))
            add("qaoa_ba", f"qaoa_ba_n{n}_m2_p1_s{seed}", n, 1, seed, qaoa_barabasi_albert(n, 2, 1, seed))
            add("random_ct", f"random_ct_n{n}_d8_s{seed}", n, 8, seed, random_ct(n, 8, seed))
            add("random_interaction_ct", f"random_interaction_ct_n{n}_d6_s{seed}", n, 6, seed, random_interaction_ct(n, 6, seed, p_edge=0.16))
            add("phase_poly", f"phase_poly_n{n}_t{3*n}_s{seed}", n, 3*n, seed, phase_polynomial_stress(n, 3*n, seed))
            add("iqp_commuting", f"iqp_commuting_n{n}_t{3*n}_s{seed}", n, 3*n, seed, commuting_iqp(n, 3*n, seed))
            if include_random_cnot_t_large or n <= max_random_cnot_t_n:
                add("random_cnot_t", f"random_cnot_t_n{n}_l{3*n}_s{seed}", n, 3*n, seed, random_cnot_t_layered(n, 3*n, seed))
            add("bv_phase_like", f"bv_phase_like_n{n}_s{seed}", n+1, n, seed, bv_oracle_like(n, seed))
    return specs

# ---------------------------------------------------------------------------
# QASM and circuit helpers
# ---------------------------------------------------------------------------

def circuit_to_qasm(qc: QuantumCircuit) -> str:
    """Export circuits using OpenQASM 2 only.

    The pipeline avoids QASM3. The experiment is about ZX/SAT
    representation sensitivity, not about optional QASM3 parser availability.
    """
    return circuit_to_qasm2_strict(qc)


def circuit_from_qasm(txt: str) -> QuantumCircuit:
    """Load OpenQASM 2 only. OpenQASM 3 inputs are skipped upstream."""
    return circuit_from_qasm2_strict(txt)


def circuit_to_qasm2_strict(qc: QuantumCircuit) -> str:
    """Return OpenQASM 2 only. Used for PyZX/tket interoperability.

    The publication experiments are not about QASM3 support. Several tool
    chains still invoke optional QASM3 importers internally, which can make
    normalisation appear to fail for environmental rather than mathematical
    reasons. PyZX strategies therefore use this strict QASM2 path.
    """
    if qasm2_dumps is not None:
        return qasm2_dumps(qc)
    # Older Qiskit exposes QuantumCircuit.qasm(); keep as fallback.
    if hasattr(qc, "qasm"):
        return qc.qasm()
    raise RuntimeError("QASM2 export unavailable; install qiskit with qasm2 support")


def sanitize_qasm2_text(txt: str) -> str:
    """Normalize PyZX-produced OpenQASM into a conservative OpenQASM 2 string.

    This function deliberately does not support OpenQASM 3.  The normalisation
    sensitivity experiments should not depend on QASM3 optional importers.
    """
    t = str(txt or "")
    # PyZX normally emits QASM2, but some versions omit qelib1 include.
    if "OPENQASM 3" in t:
        raise RuntimeError("QASM3 text encountered on strict QASM2 path")
    if "OPENQASM 2" not in t.splitlines()[:3]:
        t = "OPENQASM 2.0;\n" + t
    if 'include "qelib1.inc";' not in t and 'include "stdgates.inc";' not in t:
        lines = t.splitlines()
        # insert after OPENQASM header
        inserted = False
        for i, line in enumerate(lines):
            if line.strip().startswith("OPENQASM 2"):
                lines.insert(i + 1, 'include "qelib1.inc";')
                inserted = True
                break
        if not inserted:
            lines.insert(0, 'include "qelib1.inc";')
        t = "\n".join(lines)

    # PyZX sometimes emits `swap a,b;` in otherwise OpenQASM 2 text.
    # Qiskit's strict qasm2 loader may reject it depending on the included
    # basis/library.  Expand SWAP into three CNOTs before parsing.  This is
    # semantics-preserving and only affects serialization, not the analysis
    # logic.  Handles both `swap q[0],q[1];` and whitespace variants.
    def _expand_swap(m):
        a = m.group(1).strip()
        b = m.group(2).strip()
        return f"cx {a},{b};\ncx {b},{a};\ncx {a},{b};"
    t = re.sub(r"(?im)^\s*swap\s+([^,;]+)\s*,\s*([^;]+?)\s*;\s*$", _expand_swap, t)

    # Some PyZX/QASM writers emit `phase(...)` or `u1(...)`; qelib1 supports
    # u1 but not always phase in strict loaders.  Keep this conservative.
    t = re.sub(r"(?i)\bphase\s*\(", "u1(", t)
    return t


def circuit_from_qasm2_strict(txt: str) -> QuantumCircuit:
    """Load OpenQASM 2 only. Used for outputs produced by PyZX.

    Important: this function never calls QuantumCircuit.from_qasm_str because
    recent Qiskit versions may internally route that helper through the optional
    qiskit-qasm3-import package.  For the publication run we want PyZX failures
    to be genuine extraction/normalisation failures, not QASM3 environment noise.
    """
    t = sanitize_qasm2_text(txt)
    if qasm2_loads is None:
        raise RuntimeError("QASM2 loader unavailable; cannot use strict PyZX path")
    return qasm2_loads(t)


def save_qasm(qc: QuantumCircuit, path: str | Path) -> None:
    text_write(path, circuit_to_qasm(qc))


def transpile_ct(qc: QuantumCircuit, opt: int = 1) -> QuantumCircuit:
    basis = ["h", "x", "z", "s", "sdg", "t", "tdg", "cx", "ccx", "rz", "rx", "ry"]
    try:
        return transpile(qc, basis_gates=basis, optimization_level=opt)
    except Exception:
        try:
            return transpile(qc, optimization_level=opt)
        except Exception:
            return qc.copy()


def gate_histogram(qc: QuantumCircuit) -> Dict[str, int]:
    c = Counter()
    for inst in qc.data:
        c[inst.operation.name.lower()] += 1
    return dict(sorted(c.items()))


def qid_map(qc: QuantumCircuit) -> Dict[Any, int]:
    return {q: i for i, q in enumerate(qc.qubits)}


def angle_to_tlike(angle: float, tol: float = ANGLE_TOL_DEFAULT) -> Optional[str]:
    # Only odd multiples of pi/4 are counted as T-like. pi/2 is Clifford S-like.
    a = ((float(angle) + math.pi) % (2 * math.pi)) - math.pi
    k = int(round(a / (math.pi / 4)))
    if abs(a - k * (math.pi / 4)) <= tol and (k % 2 != 0):
        return "t" if (k % 8) in (1, 5) else "tdg"
    return None


def t_nodes(qc: QuantumCircuit, tol: float = ANGLE_TOL_DEFAULT) -> List[Tuple[int, str, List[int]]]:
    qmap = qid_map(qc)
    out = []
    for idx, inst in enumerate(qc.data):
        name = inst.operation.name.lower()
        qs = [qmap[q] for q in inst.qubits]
        if name in T_NAMES:
            out.append((idx, name, qs))
        elif name in ("rz", "p", "u1"):
            try:
                lab = angle_to_tlike(float(inst.operation.params[0]), tol=tol)
                if lab is not None:
                    out.append((idx, lab, qs))
            except Exception:
                pass
    return out


def naive_tdepth(qc: QuantumCircuit, tol: float = ANGLE_TOL_DEFAULT) -> int:
    qmap = qid_map(qc)
    last: Dict[int, int] = defaultdict(int)
    d = 0
    for inst in qc.data:
        name = inst.operation.name.lower()
        is_t = name in T_NAMES
        if not is_t and name in ("rz", "p", "u1"):
            try: is_t = angle_to_tlike(float(inst.operation.params[0]), tol=tol) is not None
            except Exception: is_t = False
        if is_t:
            qs = [qmap[q] for q in inst.qubits]
            layer = 1 + max([last[q] for q in qs] or [0])
            for q in qs: last[q] = layer
            d = max(d, layer)
    return int(d)


def controls_target(name: str, qids: List[int]) -> Tuple[List[int], Optional[int]]:
    if name in ("cx", "cy", "cz") and len(qids) >= 2:
        return [qids[0]], qids[1]
    if name in ("ccx", "ccz", "mcx", "mcx_gray", "mcx_recursive") and len(qids) >= 2:
        return qids[:-1], qids[-1]
    return [], None


# ---------------------------------------------------------------------------
# Normalization strategies
# ---------------------------------------------------------------------------

@dataclass
class NormResult:
    strategy: str
    success: bool
    qc: Optional[Any]
    error: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None


def qiskit_strategy(name: str, opt: int) -> Callable[[QuantumCircuit, int], NormResult]:
    def fn(qc: QuantumCircuit, seed: int) -> NormResult:
        try:
            out = transpile_ct(qc, opt=opt)
            out.name = f"{qc.name}_{name}"
            return NormResult(name, True, out, trace={"tool":"qiskit", "optimization_level": opt})
        except Exception as e:
            return NormResult(name, False, None, error=str(e), trace={"tool":"qiskit", "optimization_level": opt})
    setattr(fn, "strategy_name", name)
    return fn


def pyzx_strategy(name: str, mode: str, repeat: int = 1) -> Callable[[QuantumCircuit, int], NormResult]:
    def fn(qc: QuantumCircuit, seed: int) -> NormResult:
        if not HAVE_PYZX:
            return NormResult(name, False, None, error="PyZX unavailable", trace={"tool":"pyzx", "mode":mode})
        try:
            qasm = circuit_to_qasm2_strict(qc)
            c = zx.Circuit.from_qasm(qasm)
            g = c.to_graph()
            pre = {"vertices": int(g.num_vertices()), "edges": int(g.num_edges())}
            ops = []
            rnd = random.Random(seed)
            # PyZX simplification calls are version-dependent; try conservative sequences.
            for _ in range(max(1, repeat)):
                if mode == "none":
                    pass
                elif mode == "full_reduce":
                    zx.simplify.full_reduce(g); ops.append("full_reduce")
                elif mode == "clifford_simp":
                    try: zx.simplify.clifford_simp(g); ops.append("clifford_simp")
                    except Exception:
                        zx.simplify.full_reduce(g); ops.append("fallback_full_reduce")
                elif mode == "teleport_reduce":
                    try:
                        zx.simplify.teleport_reduce(g); ops.append("teleport_reduce")
                    except Exception:
                        zx.simplify.full_reduce(g); ops.append("fallback_full_reduce")
                elif mode == "pivot_only":
                    try: zx.simplify.pivot_simp(g); ops.append("pivot_simp")
                    except Exception: pass
                elif mode == "lcomp_only":
                    try: zx.simplify.lcomp_simp(g); ops.append("lcomp_simp")
                    except Exception: pass
                elif mode == "random_mix":
                    choices = ["pivot", "lcomp", "clifford"]
                    for _j in range(3):
                        r = rnd.choice(choices)
                        try:
                            if r == "pivot": zx.simplify.pivot_simp(g); ops.append("pivot_simp")
                            elif r == "lcomp": zx.simplify.lcomp_simp(g); ops.append("lcomp_simp")
                            else: zx.simplify.clifford_simp(g); ops.append("clifford_simp")
                        except Exception as ee:
                            ops.append(f"{r}_failed:{type(ee).__name__}")
                else:
                    zx.simplify.full_reduce(g); ops.append("default_full_reduce")
            post = {"vertices": int(g.num_vertices()), "edges": int(g.num_edges())}
            try:
                c2 = zx.extract.extract_circuit(g.copy())
            except Exception:
                c2 = zx.Circuit.from_graph(g.copy()) if hasattr(zx.Circuit, "from_graph") else None
            if c2 is None:
                raise RuntimeError("PyZX extraction failed")
            out = None
            conversion_errors = []
            # Prefer direct PyZX -> Qiskit conversion if available; it avoids all QASM importers.
            for conv_name, conv in [
                ("to_basic_gates_to_qiskit", lambda cc: cc.to_basic_gates().to_qiskit()),
                ("to_qiskit", lambda cc: cc.to_qiskit()),
            ]:
                try:
                    out = conv(c2)
                    ops.append(f"conversion:{conv_name}")
                    break
                except Exception as ce:
                    conversion_errors.append(f"{conv_name}:{type(ce).__name__}:{ce}")
            if out is None:
                try:
                    qasm2 = c2.to_qasm()
                    qasm2 = sanitize_qasm2_text(qasm2)
                    out = circuit_from_qasm2_strict(qasm2)
                    ops.append("conversion:qasm2_strict")
                except Exception as ce:
                    conversion_errors.append(f"qasm2_strict:{type(ce).__name__}:{ce}")
                    raise RuntimeError("PyZX-to-Qiskit conversion failed without QASM3 path: " + " | ".join(conversion_errors))
            out = transpile_ct(out, opt=0)
            out.name = f"{qc.name}_{name}"
            return NormResult(name, True, out, trace={"tool":"pyzx", "mode":mode, "ops":ops, "pre":pre, "post":post})
        except Exception as e:
            return NormResult(name, False, None, error=repr(e), trace={"tool":"pyzx", "mode":mode, "traceback": traceback.format_exc()[-2000:]})
    setattr(fn, "strategy_name", name)
    return fn


def tket_strategy(name: str) -> Callable[[QuantumCircuit, int], NormResult]:
    def fn(qc: QuantumCircuit, seed: int) -> NormResult:
        if not HAVE_TKET:
            return NormResult(name, False, None, error="tket unavailable", trace={"tool":"tket", "import_error": TKET_IMPORT_ERROR})
        try:
            tk = qiskit_to_tk(qc)
            pass_name, pass_obj = _select_tket_pass()
            if pass_obj is not None:
                pass_obj.apply(tk)
            else:
                pass_name = "conversion_only"
            
            # pytket may leave implicit wire swaps/permutations after optimization.
            # Make them explicit before conversion where supported, then request
            # replace_implicit_swaps=True on newer pytket-qiskit versions.
            try:
                if hasattr(tk, "replace_implicit_wire_swaps"):
                    tk.replace_implicit_wire_swaps()
            except Exception:
                pass
            try:
                out = tk_to_qiskit(tk, replace_implicit_swaps=True)
            except TypeError:
                out = tk_to_qiskit(tk)
            out = transpile_ct(out, opt=0)
            out.name = f"{qc.name}_{name}"
            return NormResult(name, True, out, trace={"tool":"tket", "pass":pass_name, "available_passes": TKET_AVAILABLE_PASSES[:80]})
        except Exception as e:
            return NormResult(name, False, None, error=repr(e), trace={"tool":"tket", "available_passes": TKET_AVAILABLE_PASSES[:80], "traceback": traceback.format_exc()[-2000:]})
    setattr(fn, "strategy_name", name)
    return fn


def failure_category(strategy: str, error: Optional[str], trace: Optional[Dict[str, Any]] = None) -> str:
    """Classify failures explicitly for summary health reports."""
    e = (error or "").lower()
    st = (strategy or "").lower()
    if not e:
        return "none"
    if "timeout" in e or "timeoutexpired" in e:
        return "timeout"
    if "qiskit_qasm3_import" in e or "qasm3" in e or "qasm" in e:
        return "serialization_failed"
    if "unavailable" in e or "not installed" in e or "missingoptional" in e:
        return "optional_backend_missing"
    if "tket unavailable" in e or ("tket" in st and "unavailable" in e):
        return "optional_backend_missing"
    if "extract" in e or "from_graph" in e:
        return "extraction_failed"
    if "pyzx" in st or "simpl" in e or "pivot" in e or "lcomp" in e:
        return "normalization_failed"
    return "analysis_failed"


def draw_dependency_graph_safe(G: nx.Graph, path: Path, max_nodes: int = 120) -> None:
    """Small visual artifact for counterexample inspection."""
    if not HAVE_MPL or G.number_of_nodes() > max_nodes:
        return
    try:
        ensure_dir(path.parent)
        fig = plt.figure(figsize=(5.5, 4.2))
        pos = nx.spring_layout(G, seed=42) if G.number_of_nodes() > 0 else {}
        nx.draw_networkx_nodes(G, pos, node_size=160)
        nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=7)
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def build_strategies(random_rewrite_seeds: Sequence[int], enable_tket: bool = False) -> List[Callable[[QuantumCircuit, int], NormResult]]:
    strategies = [
        qiskit_strategy("qiskit_opt0", 0),
        qiskit_strategy("qiskit_opt1", 1),
        qiskit_strategy("qiskit_opt2", 2),
        qiskit_strategy("qiskit_opt3", 3),
        pyzx_strategy("pyzx_no_simp", "none"),
        pyzx_strategy("pyzx_full_reduce", "full_reduce"),
        pyzx_strategy("pyzx_clifford_simp", "clifford_simp"),
        pyzx_strategy("pyzx_teleport_reduce", "teleport_reduce"),
        pyzx_strategy("pyzx_pivot_only", "pivot_only"),
        pyzx_strategy("pyzx_lcomp_only", "lcomp_only"),
    ]
    # tket is scientifically useful as an external compiler-normalization
    # baseline, but V7.1 makes it explicit. This avoids silent 0%-success
    # strategy rows when pytket is not available in the active environment.
    if enable_tket:
        if HAVE_TKET:
            pass_name, _pass_obj = _select_tket_pass()
            print(f"[tket] enabled; selected pass: {pass_name or 'conversion_only'}")
            strategies.append(tket_strategy("tket_compiler_norm"))
        else:
            print(f"[tket] --enable-tket was requested, but import failed: {TKET_IMPORT_ERROR}")
    for s in random_rewrite_seeds:
        strategies.append(pyzx_strategy(f"pyzx_random_mix_s{s}", "random_mix", repeat=2))
    return strategies

# ---------------------------------------------------------------------------
# Dependency graph and provenance
# ---------------------------------------------------------------------------

@dataclass
class EdgeRecord:
    u: int
    v: int
    reason: str
    blocker: str
    blocker_index: int
    shared_qubit: int


def build_dep_graph(qc: QuantumCircuit, tol: float = ANGLE_TOL_DEFAULT) -> Tuple[nx.Graph, List[EdgeRecord]]:
    qmap = qid_map(qc)
    nodes = t_nodes(qc, tol=tol)
    G = nx.Graph()
    for k, (idx, name, qs) in enumerate(nodes):
        G.add_node(k, circ_idx=int(idx), gate=name, qubits=tuple(qs))
    idx2node = {idx: i for i, (idx, _, _) in enumerate(nodes)}
    ops = []
    for inst in qc.data:
        ops.append((inst.operation.name.lower(), [qmap[q] for q in inst.qubits], inst))
    records: List[EdgeRecord] = []
    for q in range(qc.num_qubits):
        seq = sorted([idx for (idx, _, qs) in nodes if q in qs])
        # Adjacent T-like gates on the same wire are separated by the intermediate operations.
        for a, b in zip(seq, seq[1:]):
            na, nb = idx2node[a], idx2node[b]
            unsafe = False
            reason, blocker, bi = "", "", -1
            for i in range(a + 1, b):
                name, qs, _inst = ops[i]
                if q not in qs:
                    continue
                if name in Z_PHASE_NAMES:
                    continue
                if name in ("cx", "cy", "cz", "ccx", "ccz", "mcx", "mcx_gray", "mcx_recursive"):
                    ctrls, tgt = controls_target(name, qs)
                    if tgt == q:
                        unsafe = True; reason = "target_line_controlled_gate"; blocker = name; bi = i; break
                    else:
                        # T/Z phase on a control line commutes with controlled-X-like operations in this abstraction.
                        continue
                unsafe = True; reason = "same_wire_noncommuting_gate"; blocker = name; bi = i; break
            if unsafe:
                G.add_edge(na, nb, reason=reason, blocker=blocker, blocker_index=bi, shared_qubit=q)
                records.append(EdgeRecord(na, nb, reason, blocker, bi, q))
    return G, records

# ---------------------------------------------------------------------------
# SAT coloring
# ---------------------------------------------------------------------------

@dataclass
class SatResult:
    Lstar: Optional[int]
    assignment: Dict[int, int]
    sat_calls: int
    vars: int
    clauses: int
    time_s: float
    trials: List[Dict[str, Any]]
    conflicts: int = 0
    decisions: int = 0
    propagations: int = 0
    restarts: int = 0
    error: Optional[str] = None


def sat_color(G: nx.Graph, L: int) -> Tuple[bool, Dict[int, int], int, int, Dict[str, int]]:
    if not HAVE_PYSAT:
        raise RuntimeError("python-sat unavailable")
    cnf = CNF()
    var: Dict[Tuple[int, int], int] = {}
    def vv(v: int, c: int) -> int:
        key = (int(v), int(c))
        if key not in var:
            var[key] = len(var) + 1
        return var[key]
    for v in G.nodes():
        cnf.append([vv(v, c) for c in range(1, L + 1)])
        for c1 in range(1, L + 1):
            for c2 in range(c1 + 1, L + 1):
                cnf.append([-vv(v, c1), -vv(v, c2)])
    for u, v in G.edges():
        for c in range(1, L + 1):
            cnf.append([-vv(u, c), -vv(v, c)])
    with Minisat22(bootstrap_with=cnf.clauses) as solver:
        ok = bool(solver.solve())
        stats = solver.accum_stats() or {}
        stat_pack = {k:int(stats.get(k, 0) or 0) for k in ["conflicts", "decisions", "propagations", "restarts"]}
        if not ok:
            return False, {}, len(var), len(cnf.clauses), stat_pack
        model = set(x for x in solver.get_model() if x > 0)
    assignment = {}
    for v in G.nodes():
        for c in range(1, L + 1):
            if vv(v, c) in model:
                assignment[int(v)] = int(c); break
    return True, assignment, len(var), len(cnf.clauses), stat_pack


def clique_lb(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0: return 0
    try:
        return max((len(c) for c in nx.find_cliques(G)), default=1)
    except Exception:
        return 1


def greedy_ub(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0: return 0
    col = nx.coloring.greedy_color(G, strategy="largest_first")
    return 1 + max(col.values(), default=-1)


def compute_Lstar(G: nx.Graph) -> SatResult:
    t0 = time.perf_counter()
    if G.number_of_nodes() == 0:
        return SatResult(0, {}, 0, 0, 0, 0.0, [])
    try:
        lb = clique_lb(G); ub = max(lb, greedy_ub(G))
        bestL, bestAsg = None, {}
        trials = []
        calls, last_vars, last_clauses = 0, 0, 0
        tot_conflicts = tot_decisions = tot_propagations = tot_restarts = 0
        lo, hi = lb, ub
        while lo <= hi:
            mid = (lo + hi) // 2
            calls += 1
            ok, asg, nv, nc, st = sat_color(G, mid)
            last_vars, last_clauses = nv, nc
            tot_conflicts += int(st.get("conflicts", 0)); tot_decisions += int(st.get("decisions", 0))
            tot_propagations += int(st.get("propagations", 0)); tot_restarts += int(st.get("restarts", 0))
            trials.append({"L": mid, "sat": bool(ok), "vars": nv, "clauses": nc, **st})
            if ok:
                bestL, bestAsg = mid, asg
                hi = mid - 1
            else:
                lo = mid + 1
        return SatResult(bestL, bestAsg, calls, last_vars, last_clauses, time.perf_counter() - t0, trials, tot_conflicts, tot_decisions, tot_propagations, tot_restarts)
    except Exception as e:
        return SatResult(None, {}, 0, 0, 0, time.perf_counter() - t0, [], error=repr(e))

# ---------------------------------------------------------------------------
# Graph invariants and spectral analysis
# ---------------------------------------------------------------------------

def canonical_graph_hash(G: nx.Graph) -> str:
    # Conservative WL hash + degree sequence + edge count. Use pynauty if available.
    try:
        if HAVE_PYNAUTY:
            n = G.number_of_nodes()
            adj = {int(i): sorted(int(j) for j in G.neighbors(i)) for i in G.nodes()}
            pg = pynauty.Graph(number_of_vertices=n, adjacency_dict=adj, directed=False)
            cert = pynauty.certificate(pg)
            return hashlib.sha256(cert).hexdigest()
    except Exception:
        pass
    try:
        h = nx.weisfeiler_lehman_graph_hash(G)
    except Exception:
        h = "wl_unavailable"
    deg = ",".join(map(str, sorted([d for _, d in G.degree()])))
    base = f"n={G.number_of_nodes()}|m={G.number_of_edges()}|wl={h}|deg={deg}"
    return sha256_text(base)


def graph_spectrum(G: nx.Graph, k: int = 8) -> Dict[str, Any]:
    n = G.number_of_nodes()
    if n == 0:
        return {"adj_spectral_radius":0.0, "lap_lambda2":0.0, "lap_spectral_radius":0.0, "spectrum_entropy":0.0, "top_adj_eigs":[]}
    try:
        A = nx.to_numpy_array(G, dtype=float)
        eigs = np.linalg.eigvalsh(A)
        abs_eigs = np.sort(np.abs(eigs))[::-1]
        spectral_radius = float(abs_eigs[0]) if len(abs_eigs) else 0.0
        L = nx.laplacian_matrix(G).toarray().astype(float)
        leigs = np.sort(np.linalg.eigvalsh(L))
        lam2 = float(leigs[1]) if len(leigs) > 1 else 0.0
        lap_radius = float(leigs[-1]) if len(leigs) else 0.0
        vals = np.abs(eigs)
        s = float(vals.sum())
        if s > 0:
            p = vals / s
            ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        else:
            ent = 0.0
        return {
            "adj_spectral_radius": spectral_radius,
            "lap_lambda2": lam2,
            "lap_spectral_radius": lap_radius,
            "spectrum_entropy": ent,
            "top_adj_eigs": [float(x) for x in np.sort(eigs)[::-1][:k]],
        }
    except Exception as e:
        return {"adj_spectral_radius":None, "lap_lambda2":None, "lap_spectral_radius":None, "spectrum_entropy":None, "top_adj_eigs":[], "spectrum_error":repr(e)}


def graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    n = G.number_of_nodes(); m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    out: Dict[str, Any] = {
        "g_n": int(n), "g_m": int(m),
        "g_density": float(nx.density(G)) if n > 1 else 0.0,
        "g_maxdeg": int(max(degs, default=0)),
        "g_avgdeg": float(sum(degs) / n) if n else 0.0,
        "g_components": int(nx.number_connected_components(G)) if n else 0,
        "g_largest_component": int(max((len(c) for c in nx.connected_components(G)), default=0)) if n else 0,
        "g_is_bipartite": bool(nx.is_bipartite(G)) if n else True,
    }
    try: out["g_is_chordal"] = bool(nx.is_chordal(G)) if n else True
    except Exception: out["g_is_chordal"] = None
    try:
        tw, _ = nx.approximation.treewidth_min_fill_in(G)
        out["g_treewidth_minfill"] = int(tw)
    except Exception:
        out["g_treewidth_minfill"] = None
    try:
        out["g_clustering_avg"] = float(nx.average_clustering(G)) if n else 0.0
    except Exception: out["g_clustering_avg"] = None
    # Degree assortativity is undefined for many degenerate dependency graphs
    # (e.g. all degrees equal, empty variance). NetworkX emits RuntimeWarning
    # in these cases; treat the value as 0.0 so long runs stay readable.
    try:
        if m > 0 and n > 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                assort = nx.degree_assortativity_coefficient(G)
            out["g_assortativity"] = float(assort) if np.isfinite(assort) else 0.0
        else:
            out["g_assortativity"] = 0.0
    except Exception:
        out["g_assortativity"] = 0.0
    out.update(graph_spectrum(G))
    out["g_hash"] = canonical_graph_hash(G)
    return out

# ---------------------------------------------------------------------------
# Equivalence guards
# ---------------------------------------------------------------------------

def equal_by_statevector(a: QuantumCircuit, b: QuantumCircuit, max_qubits: int = 10, trials: int = 16, seed: int = 0) -> Tuple[Optional[bool], str]:
    if a.num_qubits != b.num_qubits:
        return False, "different_num_qubits"
    n = a.num_qubits
    if n > max_qubits:
        # Random basis tests only; not a proof, just a guard.
        rnd = random.Random(seed)
        try:
            for _ in range(trials):
                bits = [rnd.randint(0, 1) for _ in range(n)]
                init = Statevector.from_label("".join(map(str, reversed(bits))))
                va = init.evolve(a).data
                vb = init.evolve(b).data
                if not np.allclose(va, vb, atol=1e-7):
                    return False, "random_basis_counterexample"
            return None, "random_basis_pass_not_proof"
        except Exception as e:
            return None, f"statevector_unavailable:{repr(e)}"
    try:
        # Compare operator up to global phase using statevector basis sweep.
        phase = None
        for x in range(2 ** n):
            bits = [(x >> i) & 1 for i in range(n)]
            init = Statevector.from_label("".join(map(str, reversed(bits))))
            va = init.evolve(a).data
            vb = init.evolve(b).data
            if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
                continue
            # Find a nonzero component for phase alignment.
            idx = int(np.argmax(np.abs(vb)))
            if abs(vb[idx]) > 1e-10:
                ph = va[idx] / vb[idx]
                if phase is None: phase = ph
                if not np.allclose(va, phase * vb, atol=1e-7):
                    return False, f"basis_counterexample:{x}"
            else:
                if not np.allclose(va, vb, atol=1e-7):
                    return False, f"basis_counterexample:{x}"
        return True, "full_basis_statevector"
    except Exception as e:
        return None, f"statevector_error:{repr(e)}"


def _parse_qcec_result(res) -> Tuple[Optional[bool], str]:
    """Parse MQT-QCEC result objects across several API versions.

    MQT-QCEC has changed small parts of its Python API over time. Some
    versions expose a Configuration class, while others expect a plain
    qcec.verify(circ1, circ2) call. Result fields may also be enums/strings
    rather than booleans. This parser keeps the experiment robust and records
    an explicit method tag instead of failing the whole run.
    """
    try:
        for attr in ("equivalent", "considered_equivalent", "equivalence", "result", "status"):
            if hasattr(res, attr):
                val = getattr(res, attr)
                if callable(val):
                    val = val()
                if isinstance(val, bool):
                    return bool(val), f"mqt_qcec:{attr}"
                s = str(val).strip().lower()
                # Handle enum-like values such as EquivalenceCriterion.equivalent
                # and strings such as "equivalent" / "not_equivalent".
                if any(tok in s for tok in ("not_equiv", "not equivalent", "nonequivalent", "non_equiv", "false")):
                    return False, f"mqt_qcec:{attr}:{s}"
                if any(tok in s for tok in ("equivalent", "true")):
                    return True, f"mqt_qcec:{attr}:{s}"
    except Exception as e:
        return None, f"qcec_parse_error:{repr(e)}"
    return None, "qcec_unknown_result"


def qcec_equiv(a: QuantumCircuit, b: QuantumCircuit) -> Tuple[Optional[bool], str]:
    if not HAVE_QCEC:
        return None, "qcec_unavailable"

    attempts = []

    # Newer documented API: qcec.verify(qc1, qc2), optionally with kwargs.
    try:
        attempts.append("verify_no_config")
        res = mqt_qcec.verify(a, b)
        verdict, method = _parse_qcec_result(res)
        if verdict is not None:
            return verdict, method + ":verify_no_config"
    except Exception as e:
        last_error = repr(e)
    else:
        last_error = "unknown_result"

    # Some versions expose Configuration either as mqt_qcec.Configuration or
    # in a submodule. Use it only if available.
    for cfg_getter_name in ("Configuration",):
        try:
            cfg_cls = getattr(mqt_qcec, cfg_getter_name, None)
            if cfg_cls is None:
                continue
            attempts.append("verify_with_configuration")
            cfg = cfg_cls()
            try:
                res = mqt_qcec.verify(a, b, configuration=cfg)
            except TypeError:
                res = mqt_qcec.verify(a, b, cfg)
            verdict, method = _parse_qcec_result(res)
            if verdict is not None:
                return verdict, method + ":verify_with_configuration"
        except Exception as e:
            last_error = repr(e)

    # Last resort: write temporary QASM2 files and call the documented file API.
    # This helps if the installed binding has trouble with Qiskit circuit objects.
    try:
        import tempfile
        from qiskit.qasm2 import dumps as _qasm2_dumps
        with tempfile.TemporaryDirectory() as td:
            pa = os.path.join(td, "a.qasm")
            pb = os.path.join(td, "b.qasm")
            with open(pa, "w", encoding="utf-8") as fa:
                fa.write(_qasm2_dumps(a))
            with open(pb, "w", encoding="utf-8") as fb:
                fb.write(_qasm2_dumps(b))
            attempts.append("verify_qasm_files")
            res = mqt_qcec.verify(pa, pb)
            verdict, method = _parse_qcec_result(res)
            if verdict is not None:
                return verdict, method + ":verify_qasm_files"
    except Exception as e:
        last_error = repr(e)

    return None, f"qcec_unresolved:attempts={'+'.join(attempts)};last={last_error}"


def equivalence_guard(original: QuantumCircuit, variant: QuantumCircuit, max_qubits: int, pairwise: bool, seed: int) -> Tuple[str, str]:
    # Return status in {pass, fail, unknown} and method detail.
    qv, qm = qcec_equiv(original, variant)
    if qv is True: return "pass", qm
    if qv is False: return "fail", qm
    sv, sm = equal_by_statevector(original, variant, max_qubits=max_qubits, seed=seed)
    if sv is True: return "pass", sm
    if sv is False: return "fail", sm
    return "unknown", f"{qm};{sm}"

# ---------------------------------------------------------------------------
# Per-variant analysis
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    outdir: str
    mode: str
    angle_tol: float = ANGLE_TOL_DEFAULT
    max_equiv_qubits: int = 10
    pairwise_equivalence: bool = False
    random_rewrite_seeds: Tuple[int, ...] = (0, 1, 2)
    random_seeds: Tuple[int, ...] = (0, 1, 2)
    save_graphml: bool = True
    save_qasm: bool = True
    max_instances: Optional[int] = None
    qasm_dir: Optional[str] = None
    qasm_recursive: bool = False
    qasm_limit: Optional[int] = None
    include_mqtbench: bool = False
    include_supermarq: bool = False
    enable_tket: bool = False
    include_grover: bool = False
    include_qaoa_p2_large: bool = False
    max_qaoa_p2_n: int = 10
    include_random_cnot_t_large: bool = False
    max_random_cnot_t_n: int = 10
    include_hea_large: bool = False
    max_hea_n: int = 12
    include_qaoa_rr_large: bool = False
    max_qaoa_rr_n: int = 10
    random_cnot_t_heavy_n: int = 10
    hea_heavy_n: int = 12
    skip_tket_on_medium_large_heavy: bool = True
    skip_known_heavy_cases: bool = True
    fast_strategies_only: bool = False
    external_max_qubits: int = 64
    external_max_ops: int = 200000
    large_proxy_equiv_unknown_ok: bool = True
    draw_graph_png: bool = False
    graph_png_max_nodes: int = 80
    partial_save_every: int = 5
    max_variant_seconds: int = 180
    max_grover_variant_seconds: int = 45
    resume: bool = True
    skip_timed_out_strategy_after: int = 5


def edge_provenance_summary(records: List[EdgeRecord]) -> Dict[str, Any]:
    blockers = Counter(r.blocker for r in records)
    reasons = Counter(r.reason for r in records)
    return {
        "edge_blocker_hist": dict(sorted(blockers.items())),
        "edge_reason_hist": dict(sorted(reasons.items())),
        "edge_records": [dataclasses.asdict(r) for r in records[:500]],  # cap for JSON size
    }



def strategy_provenance_features(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten a normalization trace into paper-friendly provenance features."""
    if not trace:
        return {
            "norm_tool":"unknown", "norm_mode":"unknown", "norm_ops_count":0,
            "norm_spider_delta":None, "norm_edge_delta":None,
            "norm_ops_hist_json":"{}"
        }
    tool = str(trace.get("tool", "unknown"))
    mode = str(trace.get("mode", trace.get("pass", "unknown")))
    ops = trace.get("ops", []) or []
    op_hist = Counter(str(o).split(":", 1)[0] for o in ops)
    pre, post = trace.get("pre", {}) or {}, trace.get("post", {}) or {}
    def delta(key):
        try:
            return int(post.get(key)) - int(pre.get(key))
        except Exception:
            return None
    return {
        "norm_tool": tool,
        "norm_mode": mode,
        "norm_ops_count": int(len(ops)),
        "norm_spider_delta": delta("vertices"),
        "norm_edge_delta": delta("edges"),
        "norm_ops_hist_json": json.dumps(dict(sorted(op_hist.items())), sort_keys=True),
    }


def phase_gadget_proxy_metrics(qc: QuantumCircuit, tol: float = ANGLE_TOL_DEFAULT) -> Dict[str, Any]:
    """Approximate phase-gadget statistics from the extracted Clifford+T circuit.

    This is a conservative proxy, not a full ZX phase-gadget parser.
    It tracks T-like RZ/T nodes, wire distribution, angle classes, and local CNOT
    context that often explains changes in dependency graphs after extraction.
    """
    qmap = qid_map(qc)
    tns = t_nodes(qc, tol=tol)
    per_wire = Counter()
    angle_class = Counter()
    cnot_near = 0
    h_near = 0
    ops = list(qc.data)
    for idx, name, qs in tns:
        for q in qs: per_wire[int(q)] += 1
        angle_class[str(name)] += 1
        for j in [idx-2, idx-1, idx+1, idx+2]:
            if 0 <= j < len(ops):
                nm = ops[j].operation.name.lower()
                if nm in ("cx", "cz", "ccx", "ccz", "mcx", "mcx_gray", "mcx_recursive"):
                    try:
                        if set(qmap[q] for q in ops[j].qubits) & set(qs): cnot_near += 1
                    except Exception: pass
                if nm == "h":
                    try:
                        if set(qmap[q] for q in ops[j].qubits) & set(qs): h_near += 1
                    except Exception: pass
    counts = list(per_wire.values())
    if counts:
        mean = float(sum(counts)/len(counts)); mx = int(max(counts)); mn = int(min(counts))
        var = float(sum((x-mean)**2 for x in counts)/len(counts))
    else:
        mean = mx = mn = var = 0
    # A simple phase interaction proxy: connect wires that co-occur within a small CNOT/RZ/CNOT window.
    PG = nx.Graph(); PG.add_nodes_from(range(qc.num_qubits))
    for i, inst in enumerate(ops):
        nm = inst.operation.name.lower()
        if nm in ("cx", "cz"):
            qs = [qmap[q] for q in inst.qubits]
            if len(qs) == 2:
                window = ops[max(0,i-2):min(len(ops),i+3)]
                if any(w.operation.name.lower() in Z_PHASE_NAMES for w in window):
                    PG.add_edge(qs[0], qs[1])
        elif nm in ("ccx", "ccz", "mcx", "mcx_gray", "mcx_recursive"):
            qs = [qmap[q] for q in inst.qubits]
            for a,b in itertools.combinations(qs, 2): PG.add_edge(a,b)
    try:
        pg_tw, _ = nx.algorithms.approximation.treewidth_min_fill_in(PG)
    except Exception:
        pg_tw = None
    return {
        "phase_proxy_tlike_nodes": int(len(tns)),
        "phase_proxy_active_wires": int(len(per_wire)),
        "phase_proxy_max_per_wire": mx,
        "phase_proxy_min_per_wire": mn,
        "phase_proxy_mean_per_wire": mean,
        "phase_proxy_var_per_wire": var,
        "phase_proxy_angle_hist_json": json.dumps(dict(sorted(angle_class.items())), sort_keys=True),
        "phase_proxy_t_wire_hist_json": json.dumps({str(k):int(v) for k,v in sorted(per_wire.items())}, sort_keys=True),
        "phase_proxy_cnot_neighborhood_hits": int(cnot_near),
        "phase_proxy_h_neighborhood_hits": int(h_near),
        "phase_interaction_edges": int(PG.number_of_edges()),
        "phase_interaction_density": float(nx.density(PG)) if PG.number_of_nodes() > 1 else 0.0,
        "phase_interaction_treewidth": pg_tw,
        "phase_interaction_hash": canonical_graph_hash(PG),
    }


def dependency_edge_extended_metrics(G: nx.Graph, records: List[EdgeRecord]) -> Dict[str, Any]:
    blockers = Counter(r.blocker for r in records)
    reasons = Counter(r.reason for r in records)
    shared = Counter(str(r.shared_qubit) for r in records)
    target_edges = sum(1 for r in records if r.reason == "target_line_controlled_gate")
    noncomm_edges = sum(1 for r in records if r.reason == "same_wire_noncommuting_gate")
    return {
        "dep_edges_target_line": int(target_edges),
        "dep_edges_same_wire_noncommuting": int(noncomm_edges),
        "dep_edges_controlled_fraction": float(target_edges / len(records)) if records else 0.0,
        "dep_blocker_top": blockers.most_common(1)[0][0] if blockers else "none",
        "dep_blocker_top_count": int(blockers.most_common(1)[0][1]) if blockers else 0,
        "dep_shared_wire_hist_json": json.dumps(dict(sorted(shared.items())), sort_keys=True),
        "dep_blocker_hist_json_v4": json.dumps(dict(sorted(blockers.items())), sort_keys=True),
        "dep_reason_hist_json_v4": json.dumps(dict(sorted(reasons.items())), sort_keys=True),
    }


def analyze_variant(spec: BenchmarkSpec, norm: NormResult, original_t: QuantumCircuit, cfg: RunConfig, inst_dir: Path, logp: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "family": spec.family,
        "circuit_id": spec.circuit_id,
        "n": spec.n,
        "physical_qubits": spec.physical_qubits,
        "logical_size": spec.logical_size,
        "size_param": spec.size_param,
        "seed": spec.seed,
        "strategy": norm.strategy,
        "norm_success": bool(norm.success),
        "norm_error": norm.error,
    }
    if norm.trace is not None:
        row["norm_trace_json"] = json.dumps(norm.trace, sort_keys=True, default=str)
    row.update(strategy_provenance_features(norm.trace))
    if not norm.success or norm.qc is None:
        row.update({"equiv_status":"not_run", "analysis_success":False,
                    "failure_category": failure_category(norm.strategy, norm.error, norm.trace)})
        return row
    qc = norm.qc
    try:
        qc = transpile_ct(qc, opt=0)
    except Exception:
        pass
    row["variant_qubits"] = qc.num_qubits
    row["variant_ops"] = len(qc.data)
    row["variant_depth"] = int(qc.depth() or 0)
    row["input_size_class_pre"] = size_class_from_counts(qc.num_qubits, None, len(qc.data))
    hist = gate_histogram(qc)
    row["gate_hist_json"] = json.dumps(hist, sort_keys=True)
    row["cx_count"] = hist.get("cx", 0)
    row["ccx_count"] = hist.get("ccx", 0)
    row["h_count"] = hist.get("h", 0)
    row["rz_count"] = hist.get("rz", 0)
    row.update(phase_gadget_proxy_metrics(qc, tol=cfg.angle_tol))

    eq_status, eq_method = equivalence_guard(original_t, qc, cfg.max_equiv_qubits, cfg.pairwise_equivalence, seed=spec.seed)
    row["equiv_status"] = eq_status
    row["equiv_method"] = eq_method
    # Only analyze pass/unknown. Evidence claims later only use pass.
    if eq_status == "fail":
        row.update({"analysis_success":False, "failure_category":"equivalence_failed"})
        return row

    try:
        if cfg.save_qasm:
            save_qasm(qc, inst_dir / f"{safe_name(norm.strategy)}.qasm")
        G, edges = build_dep_graph(qc, tol=cfg.angle_tol)
        if cfg.save_graphml:
            try:
                write_graphml_safe(G, inst_dir / f"{safe_name(norm.strategy)}.graphml")
            except Exception as e:
                append_log(logp, f"GraphML export failed after sanitization {spec.circuit_id}/{norm.strategy}: {e}")
            if cfg.draw_graph_png and G.number_of_nodes() <= cfg.graph_png_max_nodes:
                try:
                    draw_dependency_graph_safe(G, inst_dir / f"{safe_name(norm.strategy)}_depgraph.png")
                except Exception:
                    pass
        gm = graph_metrics(G)
        sat = compute_Lstar(G)
        prov = edge_provenance_summary(edges)
        prov_ext = dependency_edge_extended_metrics(G, edges)
        json_dump(inst_dir / f"{safe_name(norm.strategy)}_graph_metrics.json", {"graph_metrics": gm, "sat": dataclasses.asdict(sat), "edge_provenance": prov, "edge_provenance_extended": prov_ext})
        row.update(gm)
        row.update(prov_ext)
        _tc = len(t_nodes(qc, tol=cfg.angle_tol))
        row.update({
            "analysis_success": True,
            "t_count": _tc,
            "size_class": size_class_from_counts(qc.num_qubits, _tc, len(qc.data)),
            "naive_tdepth": naive_tdepth(qc, tol=cfg.angle_tol),
            "clique_lb": clique_lb(G),
            "greedy_ub": greedy_ub(G),
            "Lstar": sat.Lstar,
            "sat_assignment_json": json.dumps(sat.assignment, sort_keys=True),
            "sat_calls": sat.sat_calls,
            "sat_vars": sat.vars,
            "sat_clauses": sat.clauses,
            "sat_clause_var_ratio": float(sat.clauses / sat.vars) if sat.vars else 0.0,
            "sat_time_s": sat.time_s,
            "sat_conflicts": sat.conflicts,
            "sat_decisions": sat.decisions,
            "sat_propagations": sat.propagations,
            "sat_restarts": sat.restarts,
            "sat_decision_per_var": float(sat.decisions / sat.vars) if sat.vars else 0.0,
            "sat_propagation_per_clause": float(sat.propagations / sat.clauses) if sat.clauses else 0.0,
            "sat_error": sat.error,
            "edge_blocker_hist_json": json.dumps(prov["edge_blocker_hist"], sort_keys=True),
            "edge_reason_hist_json": json.dumps(prov["edge_reason_hist"], sort_keys=True),
        })
        return row
    except Exception as e:
        row.update({"analysis_success":False, "analysis_error":repr(e),
                    "failure_category": failure_category(norm.strategy, repr(e), norm.trace),
                    "analysis_traceback":traceback.format_exc()[-3000:]})
        return row

# ---------------------------------------------------------------------------
# Aggregation and classification
# ---------------------------------------------------------------------------

def classify_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()
    valid = df[(df["analysis_success"] == True) & (df["equiv_status"].isin(["pass"])) & (~df["Lstar"].isna())].copy()
    rows = []
    for cid, g in valid.groupby("circuit_id"):
        fam = g["family"].iloc[0]
        n = int(g["n"].iloc[0])
        Ls = sorted(set(int(x) for x in g["Lstar"].dropna().tolist()))
        hashes = sorted(set(str(x) for x in g["g_hash"].dropna().tolist()))
        tws = [x for x in g["g_treewidth_minfill"].dropna().tolist()]
        clauses = [x for x in g["sat_clauses"].dropna().tolist()]
        tcounts = [x for x in g["t_count"].dropna().tolist()]
        if len(g) < 2:
            case = "insufficient_equivalent_variants"
        elif len(Ls) > 1:
            case = "Lstar_sensitive"
        elif len(hashes) > 1:
            case = "graph_varies_Lstar_stable"
        else:
            case = "graph_and_Lstar_stable"
        rows.append({
            "circuit_id": cid,
            "family": fam,
            "n": n,
            "valid_variants": int(len(g)),
            "strategies_valid": ";".join(sorted(g["strategy"].astype(str).tolist())),
            "case_class": case,
            "min_Lstar": min(Ls) if Ls else None,
            "max_Lstar": max(Ls) if Ls else None,
            "var_Lstar": (max(Ls)-min(Ls)) if Ls else None,
            "distinct_Lstar": len(Ls),
            "distinct_graph_hashes": len(hashes),
            "min_treewidth": min(tws) if tws else None,
            "max_treewidth": max(tws) if tws else None,
            "var_treewidth": (max(tws)-min(tws)) if tws else None,
            "min_sat_clauses": min(clauses) if clauses else None,
            "max_sat_clauses": max(clauses) if clauses else None,
            "var_sat_clauses": (max(clauses)-min(clauses)) if clauses else None,
            "min_t_count": min(tcounts) if tcounts else None,
            "max_t_count": max(tcounts) if tcounts else None,
            "var_t_count": (max(tcounts)-min(tcounts)) if tcounts else None,
        })
    summary = pd.DataFrame(rows)

    fam_rows = []
    if not summary.empty:
        for fam, g in summary.groupby("family"):
            fam_rows.append({
                "family": fam,
                "circuits": int(len(g)),
                "Lstar_sensitive": int((g["case_class"] == "Lstar_sensitive").sum()),
                "graph_varies_Lstar_stable": int((g["case_class"] == "graph_varies_Lstar_stable").sum()),
                "graph_and_Lstar_stable": int((g["case_class"] == "graph_and_Lstar_stable").sum()),
                "insufficient": int((g["case_class"] == "insufficient_equivalent_variants").sum()),
                "sensitivity_rate": float((g["case_class"] == "Lstar_sensitive").mean()),
                "mean_var_Lstar": float(g["var_Lstar"].fillna(0).mean()),
                "max_var_Lstar": int(g["var_Lstar"].fillna(0).max()),
                "mean_distinct_graph_hashes": float(g["distinct_graph_hashes"].fillna(0).mean()),
                "max_var_sat_clauses": int(g["var_sat_clauses"].fillna(0).max()),
            })
    fam_summary = pd.DataFrame(fam_rows).sort_values(["sensitivity_rate", "max_var_Lstar"], ascending=False) if fam_rows else pd.DataFrame()

    # Counterexamples: choose one low-L and one high-L strategy for every sensitive circuit.
    ce_rows = []
    for _, srow in summary[summary["case_class"] == "Lstar_sensitive"].iterrows() if not summary.empty else []:
        cid = srow["circuit_id"]
        g = valid[valid["circuit_id"] == cid].copy()
        g_low = g.sort_values(["Lstar", "t_count", "sat_clauses"]).iloc[0]
        g_high = g.sort_values(["Lstar", "t_count", "sat_clauses"], ascending=[False, False, False]).iloc[0]
        ce_rows.append({
            "circuit_id": cid,
            "family": srow["family"],
            "n": srow["n"],
            "low_strategy": g_low["strategy"],
            "low_Lstar": int(g_low["Lstar"]),
            "low_t_count": int(g_low["t_count"]),
            "low_graph_hash": g_low["g_hash"],
            "high_strategy": g_high["strategy"],
            "high_Lstar": int(g_high["Lstar"]),
            "high_t_count": int(g_high["t_count"]),
            "high_graph_hash": g_high["g_hash"],
            "delta_Lstar": int(g_high["Lstar"] - g_low["Lstar"]),
            "delta_t_count": int(g_high["t_count"] - g_low["t_count"]),
            "low_sat_clauses": int(g_low["sat_clauses"]),
            "high_sat_clauses": int(g_high["sat_clauses"]),
        })
    counterexamples = pd.DataFrame(ce_rows).sort_values(["delta_Lstar", "n"], ascending=[False, True]) if ce_rows else pd.DataFrame()
    return summary, fam_summary, counterexamples



def strategy_health_report(df: pd.DataFrame) -> pd.DataFrame:
    """Backend/strategy health over all attempts, including failed ones.

    This table is intentionally not restricted to equivalence-pass variants. It
    documents optional backends and failed normalization branches so that the
    experimental evidence cannot be mistaken for cherry-picked successes.
    """
    if df.empty or "strategy" not in df.columns:
        return pd.DataFrame()
    rows = []
    for strat, g in df.groupby("strategy"):
        total = len(g)
        norm_success = int(g.get("norm_success", pd.Series([False]*total)).fillna(False).astype(bool).sum())
        analysis_success = int(g.get("analysis_success", pd.Series([False]*total)).fillna(False).astype(bool).sum())
        eq_pass = int((g.get("equiv_status", pd.Series([""]*total)).astype(str) == "pass").sum())
        eq_fail = int((g.get("equiv_status", pd.Series([""]*total)).astype(str) == "fail").sum())
        cats = Counter(g.get("failure_category", pd.Series(["none"]*total)).fillna("none").astype(str).tolist())
        rows.append({
            "strategy": strat,
            "attempts": int(total),
            "norm_success": norm_success,
            "analysis_success": analysis_success,
            "equiv_pass": eq_pass,
            "equiv_fail": eq_fail,
            "norm_success_rate": float(norm_success/total) if total else 0.0,
            "analysis_success_rate": float(analysis_success/total) if total else 0.0,
            "equiv_pass_rate": float(eq_pass/total) if total else 0.0,
            "failure_categories_json": json.dumps(dict(sorted(cats.items())), sort_keys=True),
        })
    return pd.DataFrame(rows).sort_values(["equiv_pass_rate", "analysis_success_rate", "strategy"], ascending=[False, False, True])


def export_counterexample_artifacts(outdir: Path, counterexamples: pd.DataFrame) -> None:
    """Create a small inspection counterexample bundle.

    The original per-instance artifacts remain in instances/<circuit_id>. This
    function writes an index and copies the low/high normalized QASM/GraphML/PNG
    artifacts when present.
    """
    if counterexamples is None or counterexamples.empty:
        return
    root = outdir / "counterexample_artifacts"
    ensure_dir(root)
    index = []
    for _, r in counterexamples.head(50).iterrows():
        cid = str(r.get("circuit_id"))
        cdir = root / safe_name(cid)
        ensure_dir(cdir)
        src_inst = outdir / "instances" / cid
        rec = dict(r)
        for label, strat_key in [("low", "low_strategy"), ("high", "high_strategy")]:
            strat = safe_name(str(r.get(strat_key)))
            for ext in ["qasm", "graphml", "_depgraph.png", "_graph_metrics.json"]:
                fn = f"{strat}.{ext}" if not ext.startswith("_") else f"{strat}{ext}"
                sp = src_inst / fn
                if sp.exists():
                    dp = cdir / f"{label}_{fn}"
                    try:
                        shutil.copy2(sp, dp)
                        rec[f"{label}_{ext}_artifact"] = str(dp.relative_to(outdir))
                    except Exception:
                        pass
        op = src_inst / "original_transpiled.qasm"
        if op.exists():
            try:
                shutil.copy2(op, cdir / "original_transpiled.qasm")
                rec["original_artifact"] = str((cdir / "original_transpiled.qasm").relative_to(outdir))
            except Exception:
                pass
        json_dump(cdir / "counterexample_record.json", rec)
        index.append(rec)
    pd.DataFrame(index).to_csv(root / "counterexample_artifact_index.csv", index=False)


def strategy_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "strategy" not in df.columns:
        return pd.DataFrame()
    d = df[(df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass")].copy()
    if d.empty:
        return pd.DataFrame()
    rows = []
    for strat, g in d.groupby("strategy"):
        rows.append({
            "strategy": strat,
            "variants": int(len(g)),
            "families": int(g["family"].nunique()) if "family" in g else 0,
            "median_Lstar": float(g["Lstar"].median()) if "Lstar" in g else None,
            "mean_Lstar": float(g["Lstar"].mean()) if "Lstar" in g else None,
            "median_t_count": float(g["t_count"].median()) if "t_count" in g else None,
            "median_sat_clauses": float(g["sat_clauses"].median()) if "sat_clauses" in g else None,
            "median_treewidth": float(g["g_treewidth_minfill"].median()) if "g_treewidth_minfill" in g else None,
            "mean_phase_edges": float(g["phase_interaction_edges"].mean()) if "phase_interaction_edges" in g else None,
            "mean_dep_controlled_fraction": float(g["dep_edges_controlled_fraction"].mean()) if "dep_edges_controlled_fraction" in g else None,
            "mean_sat_decisions": float(g["sat_decisions"].mean()) if "sat_decisions" in g else None,
        })
    return pd.DataFrame(rows).sort_values(["median_Lstar", "median_sat_clauses", "strategy"], ascending=[True, True, True])


def explanatory_delta_summary(df: pd.DataFrame) -> pd.DataFrame:
    """For every L*-sensitive circuit, compare the lowest-L* and highest-L* variants."""
    if df.empty:
        return pd.DataFrame()
    valid = df[(df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass") & (~df.get("Lstar", pd.Series(dtype=float)).isna())].copy()
    rows = []
    for cid, g in valid.groupby("circuit_id"):
        Ls = sorted(set(int(x) for x in g["Lstar"].dropna().tolist()))
        if len(Ls) <= 1:
            continue
        low = g.sort_values(["Lstar", "t_count", "sat_clauses"]).iloc[0]
        high = g.sort_values(["Lstar", "t_count", "sat_clauses"], ascending=[False, False, False]).iloc[0]
        def diff(col):
            try: return float(high.get(col, 0) or 0) - float(low.get(col, 0) or 0)
            except Exception: return None
        rows.append({
            "circuit_id": cid,
            "family": low.get("family"),
            "n": low.get("n"),
            "low_strategy": low.get("strategy"),
            "high_strategy": high.get("strategy"),
            "delta_Lstar": diff("Lstar"),
            "delta_t_count": diff("t_count"),
            "delta_naive_tdepth": diff("naive_tdepth"),
            "delta_sat_clauses": diff("sat_clauses"),
            "delta_sat_vars": diff("sat_vars"),
            "delta_treewidth": diff("g_treewidth_minfill"),
            "delta_edges": diff("g_edges"),
            "delta_phase_edges": diff("phase_interaction_edges"),
            "delta_phase_active_wires": diff("phase_proxy_active_wires"),
            "delta_phase_var_per_wire": diff("phase_proxy_var_per_wire"),
            "delta_dep_target_line_edges": diff("dep_edges_target_line"),
            "delta_dep_controlled_fraction": diff("dep_edges_controlled_fraction"),
            "low_graph_hash": low.get("g_hash"),
            "high_graph_hash": high.get("g_hash"),
        })
    return pd.DataFrame(rows).sort_values(["delta_Lstar", "delta_sat_clauses"], ascending=[False, False]) if rows else pd.DataFrame()


def stable_negative_cases(df: pd.DataFrame) -> pd.DataFrame:
    """List stable and graph-sensitive-but-L*-stable cases as explicit negative controls."""
    if df.empty:
        return pd.DataFrame()
    valid = df[(df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass") & (~df.get("Lstar", pd.Series(dtype=float)).isna())].copy()
    rows=[]
    for cid,g in valid.groupby("circuit_id"):
        Ls = sorted(set(int(x) for x in g["Lstar"].dropna().tolist()))
        hashes = sorted(set(str(x) for x in g["g_hash"].dropna().tolist()))
        if len(g) < 2 or len(Ls) != 1:
            continue
        case = "graph_varies_Lstar_stable" if len(hashes)>1 else "graph_and_Lstar_stable"
        rows.append({
            "circuit_id":cid, "family":g["family"].iloc[0], "n":int(g["n"].iloc[0]),
            "case_class":case, "Lstar":Ls[0], "valid_variants":len(g),
            "distinct_graph_hashes":len(hashes), "min_t_count":int(g["t_count"].min()),
            "max_t_count":int(g["t_count"].max()), "min_sat_clauses":int(g["sat_clauses"].min()),
            "max_sat_clauses":int(g["sat_clauses"].max()),
        })
    return pd.DataFrame(rows).sort_values(["case_class","family","n"]) if rows else pd.DataFrame()





# ---------------------------------------------------------------------------
# V6 mechanistic/explanatory summaries
# ---------------------------------------------------------------------------

def _json_counter_field(series: pd.Series) -> Counter:
    c = Counter()
    for x in series.dropna().astype(str).tolist():
        try:
            obj = json.loads(x)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    try: c[str(k)] += int(v)
                    except Exception: c[str(k)] += 1
        except Exception:
            continue
    return c


def rewrite_provenance_summary(df: pd.DataFrame, circuit_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rewrite/normalization provenance by family, strategy, and case class.

    This is a V6 explanatory layer: it does not merely report that a circuit is
    sensitive, but correlates sensitivity with the kinds of normalization steps
    recorded in the strategy trace (full_reduce, clifford_simp, pivot, lcomp,
    teleport_reduce, random_mix, etc.).
    """
    if df.empty or circuit_summary is None or circuit_summary.empty:
        return pd.DataFrame()
    cs = circuit_summary[["circuit_id", "case_class"]].drop_duplicates()
    d = df.merge(cs, on="circuit_id", how="left")
    d = d[(d.get("analysis_success", False) == True) & (d.get("equiv_status", "") == "pass")].copy()
    if d.empty:
        return pd.DataFrame()
    rows=[]
    for keys, g in d.groupby(["case_class", "family", "strategy", "norm_tool", "norm_mode"], dropna=False):
        case, fam, strat, tool, mode = keys
        op_counts = _json_counter_field(g.get("norm_ops_hist_json", pd.Series(dtype=str)))
        rows.append({
            "case_class": case,
            "family": fam,
            "strategy": strat,
            "norm_tool": tool,
            "norm_mode": mode,
            "variants": int(len(g)),
            "circuits": int(g["circuit_id"].nunique()),
            "mean_Lstar": float(g["Lstar"].mean()) if "Lstar" in g else None,
            "median_Lstar": float(g["Lstar"].median()) if "Lstar" in g else None,
            "mean_t_count": float(g["t_count"].mean()) if "t_count" in g else None,
            "mean_sat_clauses": float(g["sat_clauses"].mean()) if "sat_clauses" in g else None,
            "mean_graph_edges": float(g["g_edges"].mean()) if "g_edges" in g else None,
            "mean_phase_edges": float(g["phase_interaction_edges"].mean()) if "phase_interaction_edges" in g else None,
            "mean_norm_ops_count": float(g.get("norm_ops_count", pd.Series([0]*len(g))).fillna(0).mean()),
            "norm_ops_hist_total_json": json.dumps(dict(sorted(op_counts.items())), sort_keys=True),
            "mean_norm_spider_delta": float(g.get("norm_spider_delta", pd.Series(dtype=float)).dropna().mean()) if "norm_spider_delta" in g and not g["norm_spider_delta"].dropna().empty else None,
            "mean_norm_edge_delta": float(g.get("norm_edge_delta", pd.Series(dtype=float)).dropna().mean()) if "norm_edge_delta" in g and not g["norm_edge_delta"].dropna().empty else None,
        })
    return pd.DataFrame(rows).sort_values(["case_class", "family", "strategy"])


def strategy_pair_sensitivity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise strategy comparison table: how often do two strategies disagree on L*?

    This helps show that the phenomenon is not an artifact of one single strategy:
    e.g. qiskit_opt2 vs pyzx_clifford_simp disagreement across QAOA path/star/ring.
    """
    if df.empty:
        return pd.DataFrame()
    d = df[(df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass") & (~df.get("Lstar", pd.Series(dtype=float)).isna())].copy()
    if d.empty:
        return pd.DataFrame()
    rows=[]
    for cid, g in d.groupby("circuit_id"):
        fam = g["family"].iloc[0]
        vals = {str(r["strategy"]): int(r["Lstar"]) for _, r in g.iterrows()}
        for a, b in itertools.combinations(sorted(vals), 2):
            rows.append({
                "circuit_id": cid,
                "family": fam,
                "strategy_a": a,
                "strategy_b": b,
                "Lstar_a": vals[a],
                "Lstar_b": vals[b],
                "delta_abs": abs(vals[a]-vals[b]),
                "disagree": int(vals[a] != vals[b]),
            })
    if not rows:
        return pd.DataFrame()
    r = pd.DataFrame(rows)
    out = r.groupby(["family", "strategy_a", "strategy_b"]).agg(
        circuits=("circuit_id", "nunique"),
        disagreements=("disagree", "sum"),
        disagreement_rate=("disagree", "mean"),
        max_delta_abs=("delta_abs", "max"),
        mean_delta_abs=("delta_abs", "mean"),
    ).reset_index()
    return out.sort_values(["disagreement_rate", "max_delta_abs", "circuits"], ascending=[False, False, False])


def phase_mechanism_summary(df: pd.DataFrame, circuit_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare phase/dependency proxy means across stability classes.

    This table is meant for the discussion section: it gives a mechanistic clue
    about whether sensitivity correlates with phase fragmentation, controlled-edge
    fractions, CNOT neighborhoods, or graph spectral changes.
    """
    if df.empty or circuit_summary is None or circuit_summary.empty:
        return pd.DataFrame()
    cs = circuit_summary[["circuit_id", "case_class"]].drop_duplicates()
    d = df.merge(cs, on="circuit_id", how="left")
    d = d[(d.get("analysis_success", False) == True) & (d.get("equiv_status", "") == "pass")].copy()
    if d.empty:
        return pd.DataFrame()
    metric_cols = [c for c in [
        "t_count", "naive_tdepth", "Lstar", "sat_vars", "sat_clauses", "sat_clause_var_ratio",
        "sat_conflicts", "sat_decisions", "sat_propagations", "g_edges", "g_density",
        "g_treewidth_minfill", "g_clustering", "g_degree_assortativity", "adj_spectral_radius",
        "lap_lambda2", "spectrum_entropy", "phase_proxy_tlike_nodes", "phase_proxy_active_wires",
        "phase_proxy_max_per_wire", "phase_proxy_var_per_wire", "phase_proxy_cnot_neighborhood_hits",
        "phase_proxy_h_neighborhood_hits", "phase_interaction_edges", "phase_interaction_density",
        "phase_interaction_treewidth", "dep_edges_target_line", "dep_edges_same_wire_noncommuting",
        "dep_edges_controlled_fraction"
    ] if c in d.columns]
    rows=[]
    for keys, g in d.groupby(["case_class", "family"], dropna=False):
        case, fam = keys
        row = {"case_class": case, "family": fam, "variants": int(len(g)), "circuits": int(g["circuit_id"].nunique())}
        for c in metric_cols:
            try:
                row[f"mean_{c}"] = float(pd.to_numeric(g[c], errors="coerce").mean())
                row[f"median_{c}"] = float(pd.to_numeric(g[c], errors="coerce").median())
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["case_class", "family"])


def minimal_counterexample_candidates(df: pd.DataFrame, counterexamples: pd.DataFrame) -> pd.DataFrame:
    """Rank counterexamples by simplicity and evidence quality.

    This is not a full gate-deletion reducer. It ranks candidate witnesses already present in the run:
    it identifies the smallest equivalence-pass witnesses already present in the
    benchmark run, prioritizing low qubit count, low operation count, low T-count,
    and high delta in L*/SAT clauses. These are the circuits to inspect manually
    and later reduce symbolically.
    """
    if counterexamples is None or counterexamples.empty or df.empty:
        return pd.DataFrame()
    d = df[(df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass")].copy()
    rows=[]
    for _, ce in counterexamples.iterrows():
        cid = ce.get("circuit_id")
        g = d[d["circuit_id"] == cid]
        if g.empty: continue
        rows.append({
            "circuit_id": cid,
            "family": ce.get("family"),
            "n": int(ce.get("n", g["n"].iloc[0]) or 0),
            "delta_Lstar": ce.get("delta_Lstar"),
            "low_strategy": ce.get("low_strategy"),
            "high_strategy": ce.get("high_strategy"),
            "min_ops": int(pd.to_numeric(g.get("variant_ops"), errors="coerce").min()),
            "max_ops": int(pd.to_numeric(g.get("variant_ops"), errors="coerce").max()),
            "min_t_count": int(pd.to_numeric(g.get("t_count"), errors="coerce").min()),
            "max_t_count": int(pd.to_numeric(g.get("t_count"), errors="coerce").max()),
            "min_sat_clauses": int(pd.to_numeric(g.get("sat_clauses"), errors="coerce").min()),
            "max_sat_clauses": int(pd.to_numeric(g.get("sat_clauses"), errors="coerce").max()),
            "valid_variants": int(len(g)),
            "distinct_graph_hashes": int(g.get("g_hash", pd.Series(dtype=str)).nunique()),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["simplicity_score"] = (
        out["n"].fillna(9999)*1000000 + out["min_ops"].fillna(9999)*1000 + out["min_t_count"].fillna(9999)
        - out["delta_Lstar"].fillna(0)*100
    )
    return out.sort_values(["simplicity_score", "delta_Lstar"], ascending=[True, False])



# ---------------------------------------------------------------------------
# V7 mechanistic explanation layer
# ---------------------------------------------------------------------------

def _json_counter(val: Any) -> Counter:
    """Parse JSON histogram fields safely into a Counter."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return Counter()
    if isinstance(val, dict):
        return Counter(val)
    try:
        obj = json.loads(str(val))
        if isinstance(obj, dict):
            return Counter({str(k): int(v) if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v) else 1 for k, v in obj.items()})
    except Exception:
        pass
    return Counter()


def _counter_to_json(c: Counter) -> str:
    return json.dumps({str(k): int(v) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}, sort_keys=True)


def typed_dependency_summary(df: pd.DataFrame, circuit_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate edge/blocker provenance by stability class and family.

    Per-variant edge blocker/reason histograms are converted here into a
    table turns those JSON blobs into a typed-dependency summary that can be
    used directly in the mechanism section of a paper.
    """
    if df is None or df.empty or circuit_summary is None or circuit_summary.empty:
        return pd.DataFrame()
    cs = circuit_summary[["circuit_id", "case_class"]].drop_duplicates()
    d = df.merge(cs, on="circuit_id", how="left")
    d = d[(d.get("analysis_success", False) == True) & (d.get("equiv_status", "") == "pass")].copy()
    if d.empty:
        return pd.DataFrame()
    rows=[]
    for (case, fam), g in d.groupby(["case_class", "family"], dropna=False):
        blocker = Counter(); reason = Counter(); shared = Counter()
        for _, r in g.iterrows():
            blocker.update(_json_counter(r.get("edge_blocker_hist_json")))
            blocker.update(_json_counter(r.get("dep_blocker_hist_json_v4")))
            reason.update(_json_counter(r.get("edge_reason_hist_json")))
            reason.update(_json_counter(r.get("dep_reason_hist_json_v4")))
            shared.update(_json_counter(r.get("dep_shared_wire_hist_json")))
        total_edges = int(sum(blocker.values())) if blocker else int(pd.to_numeric(g.get("g_edges"), errors="coerce").fillna(0).sum())
        top_blocker = blocker.most_common(1)[0] if blocker else ("none", 0)
        top_reason = reason.most_common(1)[0] if reason else ("none", 0)
        rows.append({
            "case_class": case,
            "family": fam,
            "variants": int(len(g)),
            "circuits": int(g["circuit_id"].nunique()),
            "typed_edges_total": int(total_edges),
            "top_blocker": str(top_blocker[0]),
            "top_blocker_count": int(top_blocker[1]),
            "top_reason": str(top_reason[0]),
            "top_reason_count": int(top_reason[1]),
            "blocker_hist_json": _counter_to_json(blocker),
            "reason_hist_json": _counter_to_json(reason),
            "shared_wire_hist_json": _counter_to_json(shared),
            "mean_dep_controlled_fraction": float(pd.to_numeric(g.get("dep_edges_controlled_fraction"), errors="coerce").mean()) if "dep_edges_controlled_fraction" in g else None,
            "mean_target_line_edges": float(pd.to_numeric(g.get("dep_edges_target_line"), errors="coerce").mean()) if "dep_edges_target_line" in g else None,
            "mean_same_wire_noncommuting_edges": float(pd.to_numeric(g.get("dep_edges_same_wire_noncommuting"), errors="coerce").mean()) if "dep_edges_same_wire_noncommuting" in g else None,
        })
    return pd.DataFrame(rows).sort_values(["case_class", "family"])


def spectral_stability_summary(df: pd.DataFrame, circuit_summary: pd.DataFrame) -> pd.DataFrame:
    """Quantify whether sensitive circuits occupy distinct spectral regimes."""
    if df is None or df.empty or circuit_summary is None or circuit_summary.empty:
        return pd.DataFrame()
    cs = circuit_summary[["circuit_id", "case_class"]].drop_duplicates()
    d = df.merge(cs, on="circuit_id", how="left")
    d = d[(d.get("analysis_success", False) == True) & (d.get("equiv_status", "") == "pass")].copy()
    if d.empty:
        return pd.DataFrame()
    metrics = [c for c in [
        "adj_spectral_radius", "lap_lambda2", "lap_spectral_radius", "spectrum_entropy",
        "g_density", "g_treewidth_minfill", "g_clustering_avg", "g_assortativity",
        "phase_interaction_density", "phase_interaction_treewidth"
    ] if c in d.columns]
    rows=[]
    for (case, fam), g in d.groupby(["case_class", "family"], dropna=False):
        row={"case_class":case, "family":fam, "variants":int(len(g)), "circuits":int(g["circuit_id"].nunique())}
        for c in metrics:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if vals.empty:
                row[f"mean_{c}"] = None; row[f"std_{c}"] = None; row[f"min_{c}"] = None; row[f"max_{c}"] = None
            else:
                row[f"mean_{c}"] = float(vals.mean())
                row[f"std_{c}"] = float(vals.std(ddof=0))
                row[f"min_{c}"] = float(vals.min())
                row[f"max_{c}"] = float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["case_class", "family"])


def rewrite_causality_matrix(df: pd.DataFrame, circuit_summary: pd.DataFrame) -> pd.DataFrame:
    """Strategy/op-level causal proxy for L* changes.

    PyZX does not expose a fully verified primitive rewrite trace in all versions,
    so this is not a formal causality proof.  It is a controlled provenance proxy:
    for each L*-sensitive circuit we compare low-L* and high-L* variants and
    aggregate which tools/modes/recorded ops appear on the high side.
    """
    if df is None or df.empty or circuit_summary is None or circuit_summary.empty:
        return pd.DataFrame()
    sens_ids = set(circuit_summary.loc[circuit_summary["case_class"] == "Lstar_sensitive", "circuit_id"].astype(str))
    d = df[(df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass") & (df["circuit_id"].astype(str).isin(sens_ids))].copy()
    if d.empty:
        return pd.DataFrame()
    rows=[]
    for cid, g in d.groupby("circuit_id"):
        try:
            g = g.dropna(subset=["Lstar"])
            if g.empty or g["Lstar"].nunique() <= 1: continue
            low = g.sort_values(["Lstar", "sat_clauses", "t_count"], ascending=[True, True, True]).iloc[0]
            high = g.sort_values(["Lstar", "sat_clauses", "t_count"], ascending=[False, False, False]).iloc[0]
            low_ops = _json_counter(low.get("norm_ops_hist_json"))
            high_ops = _json_counter(high.get("norm_ops_hist_json"))
            all_ops = sorted(set(low_ops) | set(high_ops))
            op_delta = {op: int(high_ops.get(op, 0) - low_ops.get(op, 0)) for op in all_ops}
            rows.append({
                "circuit_id": cid,
                "family": high.get("family"),
                "n": high.get("n"),
                "low_strategy": low.get("strategy"),
                "high_strategy": high.get("strategy"),
                "low_tool": low.get("norm_tool"),
                "high_tool": high.get("norm_tool"),
                "low_mode": low.get("norm_mode"),
                "high_mode": high.get("norm_mode"),
                "delta_Lstar": float(high.get("Lstar", 0)) - float(low.get("Lstar", 0)),
                "delta_t_count": float(high.get("t_count", 0)) - float(low.get("t_count", 0)),
                "delta_graph_edges": float(high.get("g_edges", high.get("g_m", 0)) or 0) - float(low.get("g_edges", low.get("g_m", 0)) or 0),
                "delta_sat_clauses": float(high.get("sat_clauses", 0) or 0) - float(low.get("sat_clauses", 0) or 0),
                "delta_phase_edges": float(high.get("phase_interaction_edges", 0) or 0) - float(low.get("phase_interaction_edges", 0) or 0),
                "delta_dep_controlled_fraction": float(high.get("dep_edges_controlled_fraction", 0) or 0) - float(low.get("dep_edges_controlled_fraction", 0) or 0),
                "low_ops_hist_json": _counter_to_json(low_ops),
                "high_ops_hist_json": _counter_to_json(high_ops),
                "ops_delta_json": json.dumps(op_delta, sort_keys=True),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    pair = pd.DataFrame(rows)
    # Add an aggregate table shape by duplicating aggregate rows with circuit_id="__aggregate__".
    agg_rows=[]
    for (fam, low_tool, high_tool, high_mode), g in pair.groupby(["family", "low_tool", "high_tool", "high_mode"], dropna=False):
        op_tot = Counter()
        for v in g["ops_delta_json"]:
            op_tot.update(_json_counter(v))
        agg_rows.append({
            "circuit_id":"__aggregate__", "family":fam, "n":None,
            "low_strategy":"*", "high_strategy":"*", "low_tool":low_tool, "high_tool":high_tool,
            "low_mode":"*", "high_mode":high_mode,
            "delta_Lstar":float(g["delta_Lstar"].mean()),
            "delta_t_count":float(g["delta_t_count"].mean()),
            "delta_graph_edges":float(g["delta_graph_edges"].mean()),
            "delta_sat_clauses":float(g["delta_sat_clauses"].mean()),
            "delta_phase_edges":float(g["delta_phase_edges"].mean()),
            "delta_dep_controlled_fraction":float(g["delta_dep_controlled_fraction"].mean()),
            "low_ops_hist_json":"{}", "high_ops_hist_json":"{}", "ops_delta_json":_counter_to_json(op_tot),
            "circuits": int(g["circuit_id"].nunique())
        })
    return pd.concat([pair, pd.DataFrame(agg_rows)], ignore_index=True).sort_values(["family", "circuit_id", "delta_Lstar"], ascending=[True, True, False])


def minimal_witness_reduction_plan(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic reduction plan for the best minimal witnesses.

    The script does not perform semantics-changing gate deletion automatically,
    because that could silently create invalid witnesses.  Instead it ranks the
    smallest equivalence-pass witnesses and specifies a safe manual/symbolic
    reduction plan: preserve the two strategies, preserve equivalence, preserve
    Var(L*)>0, then greedily try removing Clifford-only regions and repeated
    commuting phase blocks.
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    rows=[]
    for _, r in candidates.head(30).iterrows():
        cid = str(r.get("circuit_id"))
        g = df[(df.get("circuit_id", "").astype(str) == cid) & (df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass")].copy() if df is not None and not df.empty else pd.DataFrame()
        strategies = sorted(g["strategy"].astype(str).unique()) if not g.empty and "strategy" in g else []
        rows.append({
            "circuit_id": cid,
            "family": r.get("family"),
            "n": r.get("n"),
            "delta_Lstar": r.get("delta_Lstar"),
            "low_strategy": r.get("low_strategy"),
            "high_strategy": r.get("high_strategy"),
            "available_strategies_json": json.dumps(strategies),
            "priority": "highest" if len(rows) < 5 else "normal",
            "reduction_objective": "minimize gates while preserving equivalence-pass and Var(L*)>0",
            "safe_reduction_steps": "1 remove terminal Clifford pairs; 2 remove repeated identical phase blocks; 3 reduce repeated QAOA layers; 4 re-run only low/high strategies; 5 verify by QCEC/statevector; 6 freeze minimal QASM witness",
        })
    return pd.DataFrame(rows)


def paper_claim_readiness_report(df: pd.DataFrame, circuit_summary: pd.DataFrame, strategy_health: pd.DataFrame) -> pd.DataFrame:
    """Small checklist table for deciding whether a run is final-run."""
    rows=[]
    valid = df[(df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass")].copy() if df is not None and not df.empty else pd.DataFrame()
    total_circuits = int(circuit_summary["circuit_id"].nunique()) if circuit_summary is not None and not circuit_summary.empty else 0
    sens = int((circuit_summary["case_class"] == "Lstar_sensitive").sum()) if circuit_summary is not None and not circuit_summary.empty else 0
    stable = int((circuit_summary["case_class"] == "graph_and_Lstar_stable").sum()) if circuit_summary is not None and not circuit_summary.empty else 0
    graph_only = int((circuit_summary["case_class"] == "graph_varies_Lstar_stable").sum()) if circuit_summary is not None and not circuit_summary.empty else 0
    rows.append({"criterion":"equivalence_pass_variants", "value":int(len(valid)), "target":">= 500 for final large run", "status":"pass" if len(valid)>=500 else "partial"})
    rows.append({"criterion":"summarized_circuits", "value":total_circuits, "target":">= 100 for final paper run", "status":"pass" if total_circuits>=100 else "partial"})
    rows.append({"criterion":"Lstar_sensitive_circuits", "value":sens, "target":">= 5 plus minimal witness", "status":"pass" if sens>=5 else "partial"})
    rows.append({"criterion":"negative_controls_stable", "value":stable, "target":">= 5", "status":"pass" if stable>=5 else "partial"})
    rows.append({"criterion":"graph_sensitive_SAT_stable", "value":graph_only, "target":">= 3", "status":"pass" if graph_only>=3 else "partial"})
    if strategy_health is not None and not strategy_health.empty:
        try:
            min_success = float(pd.to_numeric(strategy_health.get("analysis_success_rate"), errors="coerce").dropna().min())
        except Exception:
            min_success = 0.0
        rows.append({"criterion":"strategy_health_min_success_rate", "value":min_success, "target":">= 0.5 or explicitly optional", "status":"pass" if min_success>=0.5 else "inspect"})
    fams = int(valid["family"].nunique()) if not valid.empty and "family" in valid else 0
    rows.append({"criterion":"families_covered", "value":fams, "target":">= 10", "status":"pass" if fams>=10 else "partial"})
    return pd.DataFrame(rows)


def family_stability_taxonomy(circuit_summary: pd.DataFrame) -> pd.DataFrame:
    """Condense circuit-level classes into a inspection family taxonomy."""
    if circuit_summary is None or circuit_summary.empty:
        return pd.DataFrame()
    rows=[]
    for fam, g in circuit_summary.groupby("family"):
        total = len(g)
        sens = int((g["case_class"] == "Lstar_sensitive").sum())
        gvar = int((g["case_class"] == "graph_varies_Lstar_stable").sum())
        stable = int((g["case_class"] == "graph_and_Lstar_stable").sum())
        if total == 0:
            label = "empty"
        elif sens / total >= 0.5:
            label = "SAT-sensitive family"
        elif gvar / total >= 0.5:
            label = "graph-sensitive SAT-stable family"
        elif stable / total >= 0.5:
            label = "normalization-stable family"
        else:
            label = "mixed/transition family"
        rows.append({
            "family": fam, "circuits": int(total), "Lstar_sensitive": sens,
            "graph_varies_Lstar_stable": gvar, "graph_and_Lstar_stable": stable,
            "sensitivity_rate": float(sens/total) if total else 0.0,
            "graph_variation_rate": float((sens+gvar)/total) if total else 0.0,
            "taxonomy_label": label,
        })
    return pd.DataFrame(rows).sort_values(["sensitivity_rate", "graph_variation_rate", "family"], ascending=[False, False, True])


def sat_hardness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize SAT hardness metrics by family/strategy/size class."""
    if df.empty:
        return pd.DataFrame()
    d = df[(df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass")].copy()
    if d.empty: return pd.DataFrame()
    if "size_class" not in d.columns:
        d["size_class"] = "unknown"
    cols = [c for c in ["sat_vars", "sat_clauses", "sat_clause_var_ratio", "sat_time_s", "sat_conflicts", "sat_decisions", "sat_propagations", "sat_decision_per_var", "sat_propagation_per_clause"] if c in d.columns]
    agg = d.groupby(["family", "size_class", "strategy"]).agg(
        variants=("strategy", "count"), circuits=("circuit_id", "nunique"),
        median_Lstar=("Lstar", "median"), max_Lstar=("Lstar", "max"),
        median_t_count=("t_count", "median"), max_t_count=("t_count", "max"),
    ).reset_index()
    for c in cols:
        tmp = d.groupby(["family", "size_class", "strategy"])[c].agg(["median", "max", "mean"]).reset_index()
        tmp = tmp.rename(columns={"median":f"median_{c}", "max":f"max_{c}", "mean":f"mean_{c}"})
        agg = agg.merge(tmp, on=["family", "size_class", "strategy"], how="left")
    return agg.sort_values(["family", "size_class", "median_sat_clauses" if "median_sat_clauses" in agg.columns else "variants"], ascending=[True, True, False])


def scaling_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paper-grade scaling summary by family and size class."""
    if df.empty or "analysis_success" not in df.columns:
        return pd.DataFrame([])
    d = df[(df["analysis_success"] == True) & (df["equiv_status"].isin(["pass", "unknown"]))].copy()
    if d.empty: return pd.DataFrame([])
    if "size_class" not in d.columns:
        d["size_class"] = d.apply(lambda r: size_class_from_counts(int(r.get("variant_qubits", r.get("n",0)) or 0), int(r.get("t_count",0) or 0), int(r.get("variant_ops",0) or 0)), axis=1)
    agg = d.groupby(["family", "size_class"]).agg(
        variants=("strategy", "count"),
        circuits=("circuit_id", "nunique"),
        median_qubits=("variant_qubits", "median"),
        max_qubits=("variant_qubits", "max"),
        median_ops=("variant_ops", "median"),
        max_ops=("variant_ops", "max"),
        median_t_count=("t_count", "median"),
        max_t_count=("t_count", "max"),
        median_Lstar=("Lstar", "median"),
        max_Lstar=("Lstar", "max"),
        median_sat_vars=("sat_vars", "median"),
        max_sat_vars=("sat_vars", "max"),
        median_sat_clauses=("sat_clauses", "median"),
        max_sat_clauses=("sat_clauses", "max"),
        median_sat_time=("sat_time_s", "median"),
        max_sat_time=("sat_time_s", "max"),
        median_treewidth=("g_treewidth_minfill", "median"),
        max_treewidth=("g_treewidth_minfill", "max"),
    ).reset_index()
    return agg.sort_values(["family", "size_class"])


def external_benchmark_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize imported QASM/MQT/SupermarQ subsets separately."""
    if df.empty: return pd.DataFrame([])
    d = df[df["family"].isin(["external_qasm", "mqtbench", "supermarq"])].copy()
    if d.empty: return pd.DataFrame([])
    return d.groupby(["family"]).agg(
        variants=("strategy", "count"),
        circuits=("circuit_id", "nunique"),
        pass_equiv=("equiv_status", lambda s: int((s=="pass").sum())),
        unknown_equiv=("equiv_status", lambda s: int((s=="unknown").sum())),
        failed_equiv=("equiv_status", lambda s: int((s=="fail").sum())),
        median_qubits=("variant_qubits", "median"),
        max_qubits=("variant_qubits", "max"),
        median_ops=("variant_ops", "median"),
        max_ops=("variant_ops", "max"),
        median_Lstar=("Lstar", "median"),
        max_Lstar=("Lstar", "max"),
    ).reset_index()

def write_latex_tables(outdir: Path, df_variants: pd.DataFrame, circuit_summary: pd.DataFrame, family_summary: pd.DataFrame, counterexamples: pd.DataFrame) -> None:
    texdir = outdir / "latex"
    ensure_dir(texdir)
    try:
        if not family_summary.empty:
            cols = ["family", "circuits", "Lstar_sensitive", "graph_varies_Lstar_stable", "graph_and_Lstar_stable", "sensitivity_rate", "max_var_Lstar"]
            text_write(texdir / "family_summary.tex", family_summary[cols].to_latex(index=False, escape=True, float_format="%.3f"))
        if not counterexamples.empty:
            cols = ["circuit_id", "family", "n", "low_strategy", "low_Lstar", "high_strategy", "high_Lstar", "delta_Lstar", "delta_t_count"]
            text_write(texdir / "counterexamples.tex", counterexamples[cols].head(20).to_latex(index=False, escape=True))
        if not circuit_summary.empty:
            cols = ["circuit_id", "family", "n", "case_class", "valid_variants", "min_Lstar", "max_Lstar", "var_Lstar", "distinct_graph_hashes", "var_sat_clauses"]
            text_write(texdir / "circuit_summary.tex", circuit_summary[cols].head(60).to_latex(index=False, escape=True))
    except Exception as e:
        text_write(texdir / "latex_error.txt", repr(e))


def plot_outputs(outdir: Path, df: pd.DataFrame, circuit_summary: pd.DataFrame, family_summary: pd.DataFrame) -> None:
    if not HAVE_MPL:
        return
    pdir = outdir / "plots"
    ensure_dir(pdir)
    try:
        if not family_summary.empty:
            fig = plt.figure(figsize=(9, 4.8))
            x = np.arange(len(family_summary))
            plt.bar(x, family_summary["sensitivity_rate"].astype(float))
            plt.xticks(x, family_summary["family"].astype(str), rotation=45, ha="right")
            plt.ylabel("L* sensitivity rate")
            plt.title("Normalization sensitivity by circuit family")
            plt.tight_layout(); fig.savefig(pdir / "family_sensitivity_rate.pdf"); fig.savefig(pdir / "family_sensitivity_rate.png", dpi=180); plt.close(fig)
        valid = df[(df["analysis_success"] == True) & (df["equiv_status"] == "pass")].copy()
        if not valid.empty:
            fig = plt.figure(figsize=(6, 4))
            plt.scatter(valid["g_treewidth_minfill"], valid["sat_clauses"], s=18, alpha=0.7)
            plt.xlabel("Dependency graph treewidth (min-fill)")
            plt.ylabel("SAT clauses")
            plt.yscale("log")
            plt.title("SAT encoding size vs graph treewidth")
            plt.tight_layout(); fig.savefig(pdir / "treewidth_vs_sat_clauses.pdf"); fig.savefig(pdir / "treewidth_vs_sat_clauses.png", dpi=180); plt.close(fig)
            fig = plt.figure(figsize=(6, 4))
            plt.scatter(valid["t_count"], valid["Lstar"], s=18, alpha=0.7)
            plt.xlabel("T-count")
            plt.ylabel("SAT-certified L*")
            plt.title("T-count vs SAT-certified T-layering")
            plt.tight_layout(); fig.savefig(pdir / "tcount_vs_lstar.pdf"); fig.savefig(pdir / "tcount_vs_lstar.png", dpi=180); plt.close(fig)
        if not circuit_summary.empty:
            fig = plt.figure(figsize=(6, 4))
            plt.scatter(circuit_summary["distinct_graph_hashes"], circuit_summary["var_Lstar"], s=20, alpha=0.75)
            plt.xlabel("Distinct dependency graph hashes")
            plt.ylabel("Var(L*)")
            plt.title("Graph variability vs SAT-layer variability")
            plt.tight_layout(); fig.savefig(pdir / "graph_variability_vs_lstar_variability.pdf"); fig.savefig(pdir / "graph_variability_vs_lstar_variability.png", dpi=180); plt.close(fig)
    except Exception as e:
        text_write(pdir / "plot_error.txt", traceback.format_exc())

# ---------------------------------------------------------------------------
# External QASM loading
# ---------------------------------------------------------------------------


def load_qasm_specs(qasm_dir: Optional[str], limit: Optional[int] = None, recursive: bool = False,
                    max_qubits: int = 64, max_ops: int = 200000) -> List[BenchmarkSpec]:
    """Load external QASM benchmarks. Supports recursive QASM library layouts.

    This is intentionally conservative: huge files are skipped by max_qubits/max_ops
    so large external suites can be pointed at without crashing the run.
    """
    if not qasm_dir:
        return []
    specs: List[BenchmarkSpec] = []
    root = Path(qasm_dir)
    patterns = ["*.qasm", "*.qasm2"]  # V7.1: .qasm3 intentionally skipped
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(root.rglob(pat) if recursive else root.glob(pat))
    paths = sorted(set(paths))
    # Report skipped QASM3 files explicitly so the run log is transparent.
    try:
        qasm3_paths = sorted(root.rglob("*.qasm3") if recursive else root.glob("*.qasm3"))
        if qasm3_paths:
            print(f"[qasm-load] skipped {len(qasm3_paths)} .qasm3 files by strict-QASM2 policy")
    except Exception:
        pass
    if limit:
        paths = paths[:limit]
    for pth in paths:
        try:
            txt = pth.read_text(encoding="utf-8", errors="ignore")
            qc = circuit_from_qasm(txt)
            if qc.num_qubits > max_qubits or len(qc.data) > max_ops:
                print(f"[qasm-load] skipped too large {pth} q={qc.num_qubits} ops={len(qc.data)}")
                continue
            rel = safe_name(str(pth.relative_to(root)).replace(os.sep, "__"))
            specs.append(BenchmarkSpec("external_qasm", rel, qc.num_qubits, len(qc.data), 0, qc))
        except Exception as e:
            print(f"[qasm-load] failed {pth}: {e}")
    return specs


def load_mqtbench_specs(enabled: bool, mode: str, max_qubits: int = 32, max_ops: int = 200000) -> List[BenchmarkSpec]:
    """Best-effort MQTBench integration.

    The mqt.bench API has changed across releases; this function tries a few
    common call styles and silently returns an empty list if unavailable. This
    keeps the script portable while allowing SOTA benchmark inclusion when the
    package is installed.
    """
    if not enabled:
        return []
    specs: List[BenchmarkSpec] = []
    try:
        import mqt.bench as mqtbench  # type: ignore
    except Exception as e:
        print(f"[mqtbench] unavailable: {e}")
        return specs
    algos = ["ghz", "graphstate", "qft", "qftentangled", "dj", "grover-noancilla", "vqe", "qaoa"]
    qs = [4, 6, 8] if mode == "quick" else [4, 6, 8, 10, 12] if mode == "medium" else [4, 6, 8, 10, 12, 16, 20]
    for algo in algos:
        for n in qs:
            try:
                qc = None
                # Common modern API: get_benchmark(benchmark_name, level, circuit_size)
                for call in [
                    lambda: mqtbench.get_benchmark(benchmark_name=algo, level="alg", circuit_size=n),
                    lambda: mqtbench.get_benchmark(algo, "alg", n),
                    lambda: mqtbench.get_benchmark(algo, n),
                ]:
                    try:
                        qc = call(); break
                    except Exception:
                        qc = None
                if qc is None: continue
                if hasattr(qc, "decompose"):
                    try: qc = qc.decompose(reps=10)
                    except Exception: pass
                if qc.num_qubits > max_qubits or len(qc.data) > max_ops: continue
                specs.append(BenchmarkSpec("mqtbench", safe_name(f"mqt_{algo}_n{n}"), qc.num_qubits, n, 0, qc))
            except Exception:
                continue
    print(f"[mqtbench] loaded {len(specs)} circuits")
    return specs


def load_supermarq_specs(enabled: bool, mode: str, max_qubits: int = 32, max_ops: int = 200000) -> List[BenchmarkSpec]:
    """Best-effort SupermarQ integration when supermarq-benchmarks is installed."""
    if not enabled:
        return []
    specs: List[BenchmarkSpec] = []
    try:
        import supermarq  # type: ignore
        import supermarq.benchmarks as smb  # type: ignore
    except Exception as e:
        print(f"[supermarq] unavailable: {e}")
        return specs
    qs = [4, 6] if mode == "quick" else [4, 6, 8, 10] if mode == "medium" else [4, 6, 8, 10, 12, 16]
    constructors = []
    for name in ["GHZ", "MerminBell", "BitCode", "PhaseCode", "QAOAFermionicSwapProxy", "HamiltonianSimulation", "VQEProxy"]:
        if hasattr(smb, name): constructors.append((name, getattr(smb, name)))
    for name, cls in constructors:
        for n in qs:
            try:
                obj = None
                for args in [(n,), (n, 1), (n, 2)]:
                    try:
                        obj = cls(*args); break
                    except Exception:
                        obj = None
                if obj is None: continue
                qc = None
                for meth in ["circuit", "qiskit_circuit", "generate_circuit"]:
                    if hasattr(obj, meth):
                        try:
                            qc = getattr(obj, meth)(); break
                        except Exception:
                            pass
                if qc is None and hasattr(obj, "_circuit"):
                    qc = obj._circuit
                if qc is None: continue
                if qc.num_qubits > max_qubits or len(qc.data) > max_ops: continue
                specs.append(BenchmarkSpec("supermarq", safe_name(f"supermarq_{name}_n{n}"), qc.num_qubits, n, 0, qc))
            except Exception:
                continue
    print(f"[supermarq] loaded {len(specs)} circuits")
    return specs

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def is_known_heavy_case(spec: BenchmarkSpec, strat_name: str, cfg: RunConfig) -> bool:
    """Guard against backend-heavy branches that dominate runtime but are not
    needed for the main normalization-sensitivity claim.

    The heavy-case policy is conservative: it keeps the benchmark instance, but
    skips only known heavy normalization branches (mainly full/random ZX and
    tket) for families that repeatedly cause workstation-scale hangs. The skipped
    rows are explicitly marked as ``known_heavy_skipped`` and are never used as
    negative evidence.
    """
    if not getattr(cfg, "skip_known_heavy_cases", True):
        return False

    fam = str(spec.family).lower()
    cid = str(spec.circuit_id).lower()
    st = str(strat_name).lower()
    heavy_pyzx = any(x in st for x in ["full_reduce", "random_mix", "clifford_simp"])
    tket_branch = "tket" in st
    heavy_backend = heavy_pyzx or (tket_branch and getattr(cfg, "skip_tket_on_medium_large_heavy", True))

    try:
        nval = int(spec.n)
    except Exception:
        nval = 0
    try:
        ops = len(spec.circuit.data)
    except Exception:
        ops = 0

    # QAOA p=2 at n>=12 is a backend stress case; p=1 already gives the
    # separation witnesses and p=2 up to n<=10 remains in the main run.
    if fam == "qaoa_ring" and "_p2" in cid and nval > int(getattr(cfg, "max_qaoa_p2_n", 10)):
        if heavy_backend:
            return True

    # Random CNOT+T n>=10 repeatedly causes full/random ZX and tket to dominate
    # runtime. Keep fast strategies; skip only heavy branches.
    if fam == "random_cnot_t" and nval >= int(getattr(cfg, "random_cnot_t_heavy_n", 10)):
        if heavy_backend:
            return True

    # Hardware-efficient ansatz around n>=12 can be slow under tket/full ZX.
    if fam in ("hardware_efficient_linear", "hardware_efficient_circular", "real_amplitudes") and nval >= int(getattr(cfg, "hea_heavy_n", 12)):
        if heavy_backend:
            return True

    # Large decomposed arithmetic/proxy circuits: avoid only the expensive
    # random/full simplification branches, not the full instance.
    if fam in ("modular_adder_proxy", "multiplier_proxy") and ops > 800:
        if heavy_pyzx or tket_branch:
            return True

    # Generic safety: avoid expensive randomized/full ZX simplification on very
    # large decomposed circuits unless disabled explicitly.
    if ops > 2500 and heavy_pyzx:
        return True

    return False

def run(cfg: RunConfig) -> Dict[str, Any]:
    outdir = Path(cfg.outdir)
    if outdir.exists():
        # never delete by default; create subrun
        pass
    ensure_dir(outdir)
    logp = outdir / "run.log"
    append_log(logp, f"ZX normalization sensitivity start mode={cfg.mode}")
    env = {
        "python": sys.version,
        "have_qiskit": HAVE_QISKIT,
        "qiskit_error": globals().get("QISKIT_IMPORT_ERROR", None),
        "have_pyzx": HAVE_PYZX,
        "have_pysat": HAVE_PYSAT,
        "have_qcec": HAVE_QCEC,
        "have_tket": HAVE_TKET,
        "tket_import_error": globals().get("TKET_IMPORT_ERROR", None),
        "tket_available_passes": globals().get("TKET_AVAILABLE_PASSES", []),
        "tket_selected_pass": (_select_tket_pass()[0] if HAVE_TKET else None),
        "strict_qasm2_only": True,
        "have_scipy": HAVE_SCIPY,
        "have_pynauty": HAVE_PYNAUTY,
        "config": dataclasses.asdict(cfg),
    }
    json_dump(outdir / "environment.json", env)

    if not HAVE_QISKIT:
        raise RuntimeError(f"Qiskit missing: {globals().get('QISKIT_IMPORT_ERROR','unknown')}")
    if not HAVE_PYSAT:
        raise RuntimeError("python-sat is required")

    specs = make_benchmarks(cfg.mode, cfg.random_seeds, include_grover=cfg.include_grover,
                            include_qaoa_p2_large=cfg.include_qaoa_p2_large, max_qaoa_p2_n=cfg.max_qaoa_p2_n,
                            include_random_cnot_t_large=cfg.include_random_cnot_t_large, max_random_cnot_t_n=cfg.max_random_cnot_t_n,
                            include_hea_large=cfg.include_hea_large, max_hea_n=cfg.max_hea_n)
    specs += load_qasm_specs(cfg.qasm_dir, limit=cfg.qasm_limit or cfg.max_instances, recursive=cfg.qasm_recursive,
                             max_qubits=cfg.external_max_qubits, max_ops=cfg.external_max_ops)
    specs += load_mqtbench_specs(cfg.include_mqtbench, cfg.mode, max_qubits=cfg.external_max_qubits, max_ops=cfg.external_max_ops)
    specs += load_supermarq_specs(cfg.include_supermarq, cfg.mode, max_qubits=cfg.external_max_qubits, max_ops=cfg.external_max_ops)
    if cfg.max_instances is not None:
        specs = specs[:cfg.max_instances]
    append_log(logp, f"Benchmarks: {len(specs)}")

    strategies = build_strategies(cfg.random_rewrite_seeds, enable_tket=cfg.enable_tket)
    if cfg.fast_strategies_only:
        keep = []
        for st in strategies:
            nm = getattr(st, "strategy_name", getattr(st, "__name__", "strategy"))
            if nm.startswith("qiskit_") or nm in ("pyzx_no_simp", "pyzx_pivot_only", "pyzx_lcomp_only", "pyzx_teleport_reduce"):
                keep.append(st)
        strategies = keep
    strategy_names = [getattr(st, "strategy_name", getattr(st, "__name__", "strategy")) for st in strategies]
    append_log(logp, f"Strategies: {strategy_names} total={len(strategies)}")

    rows: List[Dict[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    timeout_counts: Counter[str] = Counter()
    # Resume support: reuse already completed variant rows from a previous interrupted run.
    # This is intentionally conservative and only skips rows already present for a circuit/strategy.
    if cfg.resume:
        for prev_name in ["variant_metrics_partial.csv", "variant_metrics.csv"]:
            prev_path = outdir / prev_name
            if prev_path.exists():
                try:
                    prev_df = pd.read_csv(prev_path)
                    for _, rr in prev_df.iterrows():
                        cid = str(rr.get("circuit_id", ""))
                        st = str(rr.get("strategy", ""))
                        if cid and st:
                            rows.append(rr.to_dict())
                            completed.add((cid, st))
                    append_log(logp, f"Resume loaded {len(completed)} completed variant rows from {prev_name}")
                    break
                except Exception as e:
                    append_log(logp, f"Resume load failed for {prev_name}: {e}")

    for idx, spec in enumerate(specs, start=1):
        inst_dir = outdir / "instances" / spec.circuit_id
        ensure_dir(inst_dir)
        append_log(logp, f"[{idx}/{len(specs)}] {spec.circuit_id} family={spec.family} n={spec.n}")
        original = transpile_ct(spec.circuit, opt=0)
        try:
            save_qasm(original, inst_dir / "original_transpiled.qasm")
        except Exception as e:
            append_log(logp, f"save original qasm failed: {e}")
        json_dump(inst_dir / "original_meta.json", {"family":spec.family, "n":spec.n, "ops":len(original.data), "depth":original.depth(), "gate_hist":gate_histogram(original)})
        for strat in strategies:
            strat_name = getattr(strat, "strategy_name", getattr(strat, "__name__", "strategy"))
            if (spec.circuit_id, strat_name) in completed:
                continue
            # If a strategy repeatedly times out, quarantine it for the rest of the run.
            if cfg.skip_timed_out_strategy_after and timeout_counts[strat_name] >= cfg.skip_timed_out_strategy_after:
                rows.append({
                    "family":spec.family, "circuit_id":spec.circuit_id, "n":spec.n, "size_param":spec.size_param, "seed":spec.seed,
                    "strategy":strat_name, "norm_success":False, "norm_error":"skipped_after_repeated_timeouts",
                    "analysis_success":False, "equiv_status":"not_run", "failure_category":"timeout_quarantined"
                })
                continue
            if is_known_heavy_case(spec, strat_name, cfg):
                rows.append({
                    "family":spec.family, "circuit_id":spec.circuit_id, "n":spec.n, "size_param":spec.size_param, "seed":spec.seed,
                    "strategy":strat_name, "norm_success":False, "norm_error":"skipped_known_heavy_case",
                    "analysis_success":False, "equiv_status":"not_run", "failure_category":"known_heavy_skipped",
                    "known_heavy_policy":"v7.8_workstation_safe_branch_skip"
                })
                continue
            timeout_s = cfg.max_grover_variant_seconds if str(spec.family).lower().startswith("grover") else cfg.max_variant_seconds
            try:
                with time_limit(timeout_s):
                    norm = strat(original, spec.seed)
                    row = analyze_variant(spec, norm, original, cfg, inst_dir, logp)
                rows.append(row)
            except TimeoutExpired as e:
                timeout_counts[strat_name] += 1
                append_log(logp, f"TIMEOUT {spec.circuit_id}/{strat_name}: {e}")
                rows.append({
                    "family":spec.family, "circuit_id":spec.circuit_id, "n":spec.n, "size_param":spec.size_param, "seed":spec.seed,
                    "strategy":strat_name, "norm_success":False, "norm_error":repr(e), "analysis_success":False,
                    "equiv_status":"not_run", "failure_category":"timeout", "timeout_seconds":timeout_s
                })
            except Exception as e:
                rows.append({
                    "family":spec.family, "circuit_id":spec.circuit_id, "n":spec.n, "size_param":spec.size_param, "seed":spec.seed,
                    "strategy":strat_name, "norm_success":False, "norm_error":repr(e), "analysis_success":False,
                    "equiv_status":"not_run", "failure_category":failure_category(strat_name, repr(e), None),
                    "analysis_traceback":traceback.format_exc()[-3000:]
                })
        # incremental save. Writing a large CSV after every instance becomes
        # increasingly expensive on long runs, so save periodically.
        if cfg.partial_save_every <= 1 or idx % cfg.partial_save_every == 0 or idx == len(specs):
            pd.DataFrame(rows).to_csv(outdir / "variant_metrics_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "variant_metrics.csv", index=False)
    circuit_summary, family_summary, counterexamples = classify_groups(df)
    strategy_summary = strategy_level_summary(df)
    strategy_health = strategy_health_report(df)
    explanatory_deltas = explanatory_delta_summary(df)
    negative_controls = stable_negative_cases(df)
    scaling = scaling_summary(df)
    external_summary = external_benchmark_summary(df)
    # V6 explanatory/mechanistic outputs
    rewrite_provenance = rewrite_provenance_summary(df, circuit_summary)
    strategy_pair_matrix = strategy_pair_sensitivity_matrix(df)
    phase_mechanisms = phase_mechanism_summary(df, circuit_summary)
    minimal_candidates = minimal_counterexample_candidates(df, counterexamples)
    family_taxonomy = family_stability_taxonomy(circuit_summary)
    sat_hardness = sat_hardness_summary(df)
    # V7 explanation/causality outputs
    typed_dependency = typed_dependency_summary(df, circuit_summary)
    spectral_stability = spectral_stability_summary(df, circuit_summary)
    rewrite_causality = rewrite_causality_matrix(df, circuit_summary)
    witness_reduction_plan = minimal_witness_reduction_plan(df, minimal_candidates)
    claim_readiness = paper_claim_readiness_report(df, circuit_summary, strategy_health)
    circuit_summary.to_csv(outdir / "circuit_summary.csv", index=False)
    family_summary.to_csv(outdir / "family_summary.csv", index=False)
    counterexamples.to_csv(outdir / "counterexamples.csv", index=False)
    strategy_summary.to_csv(outdir / "strategy_summary.csv", index=False)
    strategy_health.to_csv(outdir / "strategy_health.csv", index=False)
    explanatory_deltas.to_csv(outdir / "explanatory_deltas.csv", index=False)
    negative_controls.to_csv(outdir / "negative_controls.csv", index=False)
    scaling.to_csv(outdir / "scaling_summary.csv", index=False)
    external_summary.to_csv(outdir / "external_benchmark_summary.csv", index=False)
    rewrite_provenance.to_csv(outdir / "rewrite_provenance_summary.csv", index=False)
    strategy_pair_matrix.to_csv(outdir / "strategy_pair_sensitivity_matrix.csv", index=False)
    phase_mechanisms.to_csv(outdir / "phase_mechanism_summary.csv", index=False)
    minimal_candidates.to_csv(outdir / "minimal_counterexample_candidates.csv", index=False)
    family_taxonomy.to_csv(outdir / "family_stability_taxonomy.csv", index=False)
    sat_hardness.to_csv(outdir / "sat_hardness_summary.csv", index=False)
    typed_dependency.to_csv(outdir / "typed_dependency_summary.csv", index=False)
    spectral_stability.to_csv(outdir / "spectral_stability_summary.csv", index=False)
    rewrite_causality.to_csv(outdir / "rewrite_causality_matrix.csv", index=False)
    witness_reduction_plan.to_csv(outdir / "minimal_witness_reduction_plan.csv", index=False)
    claim_readiness.to_csv(outdir / "paper_claim_readiness_report.csv", index=False)

    write_latex_tables(outdir, df, circuit_summary, family_summary, counterexamples)
    plot_outputs(outdir, df, circuit_summary, family_summary)
    export_counterexample_artifacts(outdir, counterexamples)

    # Save a concise machine-readable conclusion with careful wording.
    valid = df[(df.get("analysis_success", False) == True) & (df.get("equiv_status", "") == "pass")]
    conclusion = {
        "valid_equivalent_variants": int(len(valid)),
        "circuits_total": int(len(set(df["circuit_id"]))) if not df.empty else 0,
        "circuits_summarized": int(len(circuit_summary)) if not circuit_summary.empty else 0,
        "Lstar_sensitive_circuits": int((circuit_summary["case_class"] == "Lstar_sensitive").sum()) if not circuit_summary.empty else 0,
        "graph_varies_Lstar_stable_circuits": int((circuit_summary["case_class"] == "graph_varies_Lstar_stable").sum()) if not circuit_summary.empty else 0,
        "graph_and_Lstar_stable_circuits": int((circuit_summary["case_class"] == "graph_and_Lstar_stable").sum()) if not circuit_summary.empty else 0,
        "strategy_summary_rows": int(len(strategy_summary)) if not strategy_summary.empty else 0,
        "strategy_health_rows": int(len(strategy_health)) if not strategy_health.empty else 0,
        "explanatory_delta_rows": int(len(explanatory_deltas)) if not explanatory_deltas.empty else 0,
        "negative_control_rows": int(len(negative_controls)) if not negative_controls.empty else 0,
        "scaling_summary_rows": int(len(scaling)) if not scaling.empty else 0,
        "external_benchmark_summary_rows": int(len(external_summary)) if not external_summary.empty else 0,
        "rewrite_provenance_rows": int(len(rewrite_provenance)) if not rewrite_provenance.empty else 0,
        "strategy_pair_matrix_rows": int(len(strategy_pair_matrix)) if not strategy_pair_matrix.empty else 0,
        "phase_mechanism_rows": int(len(phase_mechanisms)) if not phase_mechanisms.empty else 0,
        "minimal_counterexample_candidate_rows": int(len(minimal_candidates)) if not minimal_candidates.empty else 0,
        "family_taxonomy_rows": int(len(family_taxonomy)) if not family_taxonomy.empty else 0,
        "sat_hardness_rows": int(len(sat_hardness)) if not sat_hardness.empty else 0,
        "typed_dependency_rows": int(len(typed_dependency)) if not typed_dependency.empty else 0,
        "spectral_stability_rows": int(len(spectral_stability)) if not spectral_stability.empty else 0,
        "rewrite_causality_rows": int(len(rewrite_causality)) if not rewrite_causality.empty else 0,
        "witness_reduction_plan_rows": int(len(witness_reduction_plan)) if not witness_reduction_plan.empty else 0,
        "claim_readiness_rows": int(len(claim_readiness)) if not claim_readiness.empty else 0,
        "claim_guard": "Only variants with equivalence_status=pass are used for sensitivity claims. Unknown/failing equivalence cases are quarantined.",
        "interpretation": "If Lstar_sensitive_circuits > 0, the run provides empirical evidence that SAT-certified T-layering is normalization-relative for the tested strategies and benchmarks. It is not a proof of general non-invariance unless a minimal symbolic witness is extracted and verified separately.",
    }
    json_dump(outdir / "conclusion.json", conclusion)

    # Package key summaries.
    try:
        zpath = outdir / "summary_artifacts.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in ["variant_metrics.csv", "circuit_summary.csv", "family_summary.csv", "counterexamples.csv", "strategy_summary.csv", "strategy_health.csv", "explanatory_deltas.csv", "negative_controls.csv", "scaling_summary.csv", "external_benchmark_summary.csv", "rewrite_provenance_summary.csv", "strategy_pair_sensitivity_matrix.csv", "phase_mechanism_summary.csv", "minimal_counterexample_candidates.csv", "family_stability_taxonomy.csv", "sat_hardness_summary.csv", "typed_dependency_summary.csv", "spectral_stability_summary.csv", "rewrite_causality_matrix.csv", "minimal_witness_reduction_plan.csv", "paper_claim_readiness_report.csv", "environment.json", "conclusion.json", "run.log"]:
                p = outdir / fn
                if p.exists(): z.write(p, arcname=fn)
            if (outdir / "latex").exists():
                for p in (outdir / "latex").glob("*.tex"):
                    z.write(p, arcname=f"latex/{p.name}")
            if (outdir / "plots").exists():
                for p in (outdir / "plots").glob("*.pdf"):
                    z.write(p, arcname=f"plots/{p.name}")
    except Exception:
        append_log(logp, "zip packaging failed")
    append_log(logp, f"Done. Outputs in {outdir}")
    return conclusion


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(description="ZX normalization sensitivity experimental engine")
    ap.add_argument("--outdir", default="artifacts_normsens")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--medium", action="store_true")
    mode.add_argument("--large", action="store_true")
    ap.add_argument("--angle-tol", type=float, default=ANGLE_TOL_DEFAULT)
    ap.add_argument("--max-equiv-qubits", type=int, default=10)
    ap.add_argument("--pairwise-equivalence", action="store_true", help="Currently uses original-vs-variant equivalence guard; kept for compatibility and future pairwise matrix extension.")
    ap.add_argument("--random-rewrite-seeds", type=int, nargs="*", default=[0,1,2])
    ap.add_argument("--random-seeds", type=int, nargs="*", default=[0,1,2])
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--qasm-dir", default=None)
    ap.add_argument("--qasm-recursive", action="store_true")
    ap.add_argument("--qasm-limit", type=int, default=None)
    ap.add_argument("--include-mqtbench", action="store_true", help="Best-effort import from MQTBench if installed.")
    ap.add_argument("--include-supermarq", action="store_true", help="Best-effort import from SupermarQ if installed.")
    ap.add_argument("--enable-tket", action="store_true", help="Include tket Peephole/FullPeephole as an explicit optional compiler-normalization baseline.")
    ap.add_argument("--include-grover", action="store_true", help="Include MCX-heavy Grover-like circuits as optional stress tests. Disabled by default because they can dominate backend runtime.")
    ap.add_argument("--include-qaoa-p2-large", action="store_true", help="Include all large qaoa_ring p=2 instances. Disabled by default; p=2 is included up to --max-qaoa-p2-n only.")
    ap.add_argument("--max-qaoa-p2-n", type=int, default=10, help="Largest n for default qaoa_ring/path/star p=2 generated cases. Larger p=2 cases are optional stress tests.")
    ap.add_argument("--include-random-cnot-t-large", action="store_true", help="Include random_cnot_t instances above --max-random-cnot-t-n as optional stress tests. Disabled by default for workstation-safe publication runs.")
    ap.add_argument("--max-random-cnot-t-n", type=int, default=10, help="Largest random_cnot_t n included in the main benchmark by default. Larger instances require --include-random-cnot-t-large.")
    ap.add_argument("--include-hea-large", action="store_true", help="Include hardware-efficient ansatz instances above --max-hea-n as optional stress tests. Disabled by default for workstation-safe publication runs.")
    ap.add_argument("--max-hea-n", type=int, default=12, help="Largest hardware-efficient ansatz n included in the main benchmark by default. Larger instances require --include-hea-large.")
    ap.add_argument("--include-qaoa-rr-large", action="store_true", help="Include random-regular QAOA instances above --max-qaoa-rr-n as optional stress tests. Disabled by default for workstation-safe publication runs.")
    ap.add_argument("--max-qaoa-rr-n", type=int, default=10, help="Largest random-regular QAOA n included in the main benchmark by default. Larger instances require --include-qaoa-rr-large.")
    ap.add_argument("--random-cnot-t-heavy-n", type=int, default=10, help="From this n onward, skip only heavy random/full ZX and tket branches for random_cnot_t; fast strategies still run.")
    ap.add_argument("--hea-heavy-n", type=int, default=12, help="From this n onward, skip only heavy random/full ZX and tket branches for hardware-efficient ansatz families; fast strategies still run.")
    ap.add_argument("--do-not-skip-tket-on-heavy", action="store_true", help="Keep tket branches even on known heavy families/sizes. Not recommended for publication workstation runs.")
    ap.add_argument("--no-skip-known-heavy-cases", action="store_true", help="Disable workstation-safe skipping of known heavy backend branches.")
    ap.add_argument("--fast-strategies-only", action="store_true", help="Use only faster Qiskit/PyZX strategies for diagnostic/scaling runs.")
    ap.add_argument("--external-max-qubits", type=int, default=64)
    ap.add_argument("--external-max-ops", type=int, default=200000)
    ap.add_argument("--strict-equivalence", action="store_true", help="Do not analyze unknown-equivalence large proxy variants.")
    ap.add_argument("--no-graphml", action="store_true")
    ap.add_argument("--no-qasm", action="store_true")
    ap.add_argument("--draw-graphs", action="store_true", help="Export dependency-graph PNGs during the run. Disabled by default for speed; GraphML/JSON metrics are still written.")
    ap.add_argument("--graph-png-max-nodes", type=int, default=80)
    ap.add_argument("--partial-save-every", type=int, default=5, help="Write variant_metrics_partial.csv every N circuits instead of after every circuit.")
    ap.add_argument("--max-variant-seconds", type=int, default=180, help="Soft timeout per circuit/strategy variant. Timed-out branches are quarantined, not used for claims.")
    ap.add_argument("--max-grover-variant-seconds", type=int, default=45, help="Shorter soft timeout for Grover-like MCX-heavy variants.")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume from existing variant_metrics_partial.csv/variant_metrics.csv in outdir.")
    ap.add_argument("--skip-timed-out-strategy-after", type=int, default=5, help="Quarantine a strategy after N timeouts across the run. Use 0 to disable.")
    args = ap.parse_args()
    m = "quick" if args.quick else "large" if args.large else "medium" if args.medium else "quick"
    return RunConfig(
        outdir=args.outdir,
        mode=m,
        angle_tol=args.angle_tol,
        max_equiv_qubits=args.max_equiv_qubits,
        pairwise_equivalence=args.pairwise_equivalence,
        random_rewrite_seeds=tuple(args.random_rewrite_seeds),
        random_seeds=tuple(args.random_seeds),
        save_graphml=not args.no_graphml,
        save_qasm=not args.no_qasm,
        max_instances=args.max_instances,
        qasm_dir=args.qasm_dir,
        qasm_recursive=args.qasm_recursive,
        qasm_limit=args.qasm_limit,
        include_mqtbench=args.include_mqtbench,
        include_supermarq=args.include_supermarq,
        enable_tket=args.enable_tket,
        include_grover=args.include_grover,
        include_qaoa_p2_large=args.include_qaoa_p2_large,
        max_qaoa_p2_n=args.max_qaoa_p2_n,
        include_random_cnot_t_large=args.include_random_cnot_t_large,
        max_random_cnot_t_n=args.max_random_cnot_t_n,
        include_hea_large=args.include_hea_large,
        max_hea_n=args.max_hea_n,
        include_qaoa_rr_large=args.include_qaoa_rr_large,
        max_qaoa_rr_n=args.max_qaoa_rr_n,
        random_cnot_t_heavy_n=args.random_cnot_t_heavy_n,
        hea_heavy_n=args.hea_heavy_n,
        skip_tket_on_medium_large_heavy=(not args.do_not_skip_tket_on_heavy),
        skip_known_heavy_cases=(not args.no_skip_known_heavy_cases),
        fast_strategies_only=args.fast_strategies_only,
        external_max_qubits=args.external_max_qubits,
        external_max_ops=args.external_max_ops,
        large_proxy_equiv_unknown_ok=not args.strict_equivalence,
        draw_graph_png=args.draw_graphs,
        graph_png_max_nodes=args.graph_png_max_nodes,
        partial_save_every=max(1, args.partial_save_every),
        max_variant_seconds=args.max_variant_seconds,
        max_grover_variant_seconds=args.max_grover_variant_seconds,
        resume=not args.no_resume,
        skip_timed_out_strategy_after=args.skip_timed_out_strategy_after,
    )


if __name__ == "__main__":
    cfg = parse_args()
    try:
        conclusion = run(cfg)
        print(json.dumps(conclusion, indent=2, sort_keys=True))
    except Exception as e:
        print("FATAL:", repr(e), file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
