#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IJQI REPRODUCIBLE RESEARCH ENGINE -- PAULI ANTICOMMUTATION SENSITIVITY
==========================================================

Single-file publication implementation of the revised IJQI model.

NO command-line arguments are used.
Edit the CONFIG section and press Run in Spyder.

Purpose
-------
The same file runs the balanced benchmark study, regenerates the 4-qubit
three-path witness, computes exact five-T minimality certificates, and exports
paper-ready tables and figures. No helper Python files are required.

Pipeline
--------
    original circuit
      -> selected normalization strategy
      -> equivalence guard
      -> exact Clifford+T rebase
      -> explicit T/Tdg event extraction
      -> suffix-Clifford Pauli signatures
      -> Pauli anticommutation graph
      -> exact chromatic number
      -> exact graph-isomorphism classification
      -> T-count-matched sensitivity analysis

Scientific interpretation
-------------------------
The graph edge relation is algebraic and independent of coloring:

    {i,j} in E  <=>  <P_i,P_j>_sp = 1 (mod 2).

The chromatic number is interpreted only as:

    minimum number of mutually commuting Pauli-signature classes

for the fixed extracted/rebased representation.

It is NOT claimed to equal physical T-depth or executable schedule depth.

Design goals of this pilot
--------------------------
1. Test whether the revised abstraction is non-degenerate on real benchmarks.
2. Test whether different semantics-preserving normalizations induce
   non-isomorphic anticommutation graphs.
3. Test whether exact chromatic number varies.
4. Most importantly, test whether chromatic variation survives exact T-count
   matching between normalizations of the same original circuit.
5. Produce small witness candidates for manual inspection.
6. Keep every branch auditable: normalization trace, equivalence status,
   Pauli event table, edge witnesses, graph class, exact coloring.

V8.3.2 correction
-----------------
Zero-T representations are legitimate outcomes. Empty event/edge tables now retain fixed schemas, so such branches are analyzed as G=empty and chi=0 rather than raising KeyError('event_id'). High-level compiler outputs that leave the exact pi/4 event model are explicitly classified as `exact_CT_model_out_of_domain`; no numerical angle rounding is allowed.

Fail-closed policy
------------------
- Unsupported exact Clifford+T rebase -> branch rejected.
- Equivalence failure -> branch rejected from claims.
- Unknown equivalence -> branch retained only as diagnostic, not used in claims.
- Any core self-test failure -> entire pilot stops.
"""

from __future__ import annotations

import json
import math
import random
import hashlib
import itertools
import traceback
import re
import time
import platform
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx

# =============================================================================
# CONFIG -- edit here in Spyder
# =============================================================================

# Choose ONE profile and press F5 in Spyder.  Paper reproduces 220 x 13 branches.
# "witness" checks only the three PyZX paths; "quick" is a small smoke test.
RUN_PROFILE = "paper"             # "paper", "witness", or "quick"
RESUME_INTERRUPTED_RUN = True       # reuse completed rows only for unchanged source
SAVE_ZX_WITNESS_SNAPSHOTS = True
VERIFY_EXACT_WITNESS_CHANNEL = True
EXPORT_PAPER_FIGURES = True
EXPECTED_WITNESS_CIRCUIT = "VQE_slot07_vqe_su2_n4_r3_s7"
EXPECTED_WITNESS_CHI = {"pyzx_no_simp": 3, "pyzx_clifford_simp": 2,
                        "pyzx_pivot_only": 3}

# Output is anchored to the actual script directory, not Spyder's cwd.
_SCRIPT_HOME = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = _SCRIPT_HOME / ("zx_normalization_sensitivity_output_" + RUN_PROFILE)


# Core checks must stay enabled for scientific runs.
RUN_CORE_SELF_TESTS = True

# Save per-variant GraphML and CSV artifacts.
SAVE_GRAPHML = True
SAVE_QASM = True

# Equivalence:
MAX_OPERATOR_CHECK_QUBITS = 8
MATRIX_ATOL = 1e-9

# Targeted pilot size.
# This is designed to produce ~25-30 circuits depending on library availability.
RANDOM_SEEDS = [0, 1, 2]

# Primary analysis = PyZX strategies.
# Qiskit strategies are included as secondary compiler baselines.
INCLUDE_QISKIT_BASELINES = True

# Optional tket baseline. Leave False for first pilot unless already installed.
INCLUDE_TKET = False

# PyZX random-mix seeds.
PYZX_RANDOM_SEEDS = [0, 1, 2]

# Exact coloring cross-check with python-sat if available.
USE_SAT_CROSSCHECK = True

# Safety threshold for operator matrix equivalence.
# Above this, MQT-QCEC is used if installed; otherwise statevector fallback
# may become inconclusive rather than falsely passing.
MAX_STATEVECTOR_QUBITS = 10

# Witness shortlist size.

TOP_WITNESSES = 20

# Publication-scale balanced design: 22 predeclared families x 10 circuits.
PUBLICATION_FAMILIES = 22
CIRCUITS_PER_FAMILY = 10
EXPECTED_PUBLICATION_CIRCUITS = PUBLICATION_FAMILIES * CIRCUITS_PER_FAMILY
LOCK_BALANCED_SUITE = True

# Deep reviewer-facing validation.
# These checks are deliberately redundant: binary symplectic propagation is
# cross-checked against direct matrix conjugation, and the Pauli-rotation
# decomposition is reconstructed projectively for small circuits.
RUN_DEEP_PAULI_VALIDATION = True
MAX_DIRECT_PAULI_VALIDATION_QUBITS = 5
MAX_RECONSTRUCTION_QUBITS = 5

# A publication witness is accepted as "hand-checkable exact" only when its
# upper bound is an explicit proper coloring and its lower bound is an explicit
# clique of the same size.  SAT may still be used elsewhere as a cross-check.
REQUIRE_CLIQUE_CERTIFICATE_FOR_PRIMARY_WITNESS = True

# Store exhaustive pairwise commutation checks for small circuits only in JSON
# summary form; edge CSVs remain the compact edge-level witness.
SAVE_DIRECT_MATRIX_VALIDATION_DETAILS = True

# Reviewer-facing coverage / robustness analyses.
RUN_STRATEGY_SATURATION = True
SATURATION_PERMUTATIONS = 100
RUN_LEAVE_ONE_STRATEGY_OUT = True

# Timing / scalability accounting.
COLLECT_PERFORMANCE_METRICS = True

# All randomness used only for analysis ordering is fixed and recorded.
ANALYSIS_RANDOM_SEED = 20260820

# =============================================================================
# Imports
# =============================================================================

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator, Statevector
    try:
        from qiskit.qasm2 import dumps as qasm2_dumps, loads as qasm2_loads
    except Exception:
        qasm2_dumps = None
        qasm2_loads = None
    HAVE_QISKIT = True
except Exception as e:
    HAVE_QISKIT = False
    QISKIT_IMPORT_ERROR = repr(e)

try:
    import pyzx as zx
    HAVE_PYZX = True
except Exception as e:
    HAVE_PYZX = False
    PYZX_IMPORT_ERROR = repr(e)

try:
    from pysat.formula import CNF
    from pysat.solvers import Minisat22
    HAVE_PYSAT = True
except Exception:
    HAVE_PYSAT = False

try:
    from mqt import qcec as mqt_qcec
    HAVE_QCEC = True
except Exception:
    HAVE_QCEC = False

try:
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    import pytket.passes as _tket_passes
    HAVE_TKET = True
except Exception as e:
    HAVE_TKET = False
    TKET_IMPORT_ERROR = repr(e)

# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def json_dump(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)

def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))[:180]

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def qid_map(qc: QuantumCircuit) -> Dict[Any, int]:
    return {q: i for i, q in enumerate(qc.qubits)}

def gate_histogram(qc: QuantumCircuit) -> Dict[str, int]:
    c = Counter(inst.operation.name.lower() for inst in qc.data)
    return dict(sorted(c.items()))

# =============================================================================
# Exact Clifford+T rebase
# =============================================================================

ALLOWED_CT = {"h", "x", "z", "s", "sdg", "t", "tdg", "cx", "id", "barrier"}

def normalize_angle_pi_over_4(theta: float, tol: float = 1e-10) -> int:
    x = float(theta) / (math.pi / 4.0)
    k = int(round(x))
    if abs(x - k) > tol:
        raise ValueError(
            f"Angle {theta!r} is not an exact multiple of pi/4 within tol={tol}"
        )
    return k % 8

def append_phase_power(qc: QuantumCircuit, q: int, k_mod8: int) -> None:
    k = int(k_mod8) % 8
    if k == 0:
        return
    if k == 1:
        qc.t(q)
    elif k == 2:
        qc.s(q)
    elif k == 3:
        qc.s(q); qc.t(q)
    elif k == 4:
        qc.z(q)
    elif k == 5:
        qc.z(q); qc.t(q)
    elif k == 6:
        qc.sdg(q)
    elif k == 7:
        qc.tdg(q)

def append_ccx_7t(qc: QuantumCircuit, a: int, b: int, t: int) -> None:
    qc.h(t)
    qc.cx(b, t)
    qc.tdg(t)
    qc.cx(a, t)
    qc.t(t)
    qc.cx(b, t)
    qc.tdg(t)
    qc.cx(a, t)
    qc.t(b)
    qc.t(t)
    qc.h(t)
    qc.cx(a, b)
    qc.t(a)
    qc.tdg(b)
    qc.cx(a, b)

def exact_ct_rebase(qc: QuantumCircuit) -> QuantumCircuit:
    if not HAVE_QISKIT:
        raise RuntimeError(f"Qiskit unavailable: {globals().get('QISKIT_IMPORT_ERROR')}")

    out = QuantumCircuit(qc.num_qubits, name=f"{qc.name}_ct")
    qmap = qid_map(qc)

    for pos, inst in enumerate(qc.data):
        name = inst.operation.name.lower()
        qs = [qmap[q] for q in inst.qubits]
        pars = list(getattr(inst.operation, "params", []))

        if name == "barrier":
            out.barrier(*qs)
        elif name == "id":
            out.id(qs[0])
        elif name == "h":
            out.h(qs[0])
        elif name == "x":
            out.x(qs[0])
        elif name == "z":
            out.z(qs[0])
        elif name == "s":
            out.s(qs[0])
        elif name == "sdg":
            out.sdg(qs[0])
        elif name == "t":
            out.t(qs[0])
        elif name == "tdg":
            out.tdg(qs[0])
        elif name == "cx":
            out.cx(qs[0], qs[1])

        elif name in ("rz", "p", "u1"):
            if len(qs) != 1 or not pars:
                raise ValueError(f"Malformed {name} at position {pos}")
            k = normalize_angle_pi_over_4(float(pars[0]))
            append_phase_power(out, qs[0], k)

        elif name == "rx":
            if len(qs) != 1 or not pars:
                raise ValueError(f"Malformed rx at position {pos}")
            k = normalize_angle_pi_over_4(float(pars[0]))
            out.h(qs[0])
            append_phase_power(out, qs[0], k)
            out.h(qs[0])

        elif name == "ry":
            if len(qs) != 1 or not pars:
                raise ValueError(f"Malformed ry at position {pos}")
            k = normalize_angle_pi_over_4(float(pars[0]))
            out.sdg(qs[0])
            out.h(qs[0])
            append_phase_power(out, qs[0], k)
            out.h(qs[0])
            out.s(qs[0])

        elif name == "cz":
            out.h(qs[1])
            out.cx(qs[0], qs[1])
            out.h(qs[1])

        elif name == "swap":
            out.cx(qs[0], qs[1])
            out.cx(qs[1], qs[0])
            out.cx(qs[0], qs[1])

        elif name == "ccx":
            append_ccx_7t(out, qs[0], qs[1], qs[2])

        else:
            raise ValueError(
                f"Unsupported gate for exact Clifford+T rebase: "
                f"name={name!r}, position={pos}, qubits={qs}, params={pars}"
            )

    bad = []
    for i, inst in enumerate(out.data):
        nm = inst.operation.name.lower()
        if nm not in ALLOWED_CT:
            bad.append((i, nm))
    if bad:
        raise RuntimeError(f"Internal rebase error; non-CT gates remain: {bad[:20]}")
    return out

# =============================================================================
# Projective equivalence checks
# =============================================================================

def matrices_equal_up_to_global_phase(a: np.ndarray, b: np.ndarray,
                                      atol: float = MATRIX_ATOL) -> bool:
    if a.shape != b.shape:
        return False
    fb = b.reshape(-1)
    fa = a.reshape(-1)
    idx = int(np.argmax(np.abs(fb)))
    if abs(fb[idx]) < atol:
        return np.allclose(a, b, atol=atol)
    phase = fa[idx] / fb[idx]
    if abs(phase) < atol:
        return False
    phase = phase / abs(phase)
    return bool(np.allclose(a, phase * b, atol=atol, rtol=0))

def operator_equiv_up_to_phase(a: QuantumCircuit, b: QuantumCircuit,
                               max_qubits: int = MAX_OPERATOR_CHECK_QUBITS) -> Optional[bool]:
    if a.num_qubits != b.num_qubits:
        return False
    if a.num_qubits > max_qubits:
        return None
    return matrices_equal_up_to_global_phase(Operator(a).data, Operator(b).data)

def _parse_qcec_result(res) -> Tuple[Optional[bool], str]:
    for attr in ("equivalent", "considered_equivalent", "equivalence", "result", "status"):
        try:
            if hasattr(res, attr):
                val = getattr(res, attr)
                if callable(val):
                    val = val()
                if isinstance(val, bool):
                    return bool(val), f"mqt_qcec:{attr}"
                s = str(val).lower()
                if any(t in s for t in ("not_equiv", "not equivalent", "nonequivalent", "non_equiv", "false")):
                    return False, f"mqt_qcec:{attr}:{s}"
                if any(t in s for t in ("equivalent", "true")):
                    return True, f"mqt_qcec:{attr}:{s}"
        except Exception:
            pass
    return None, "qcec_unknown"

def qcec_equiv(a: QuantumCircuit, b: QuantumCircuit) -> Tuple[Optional[bool], str]:
    if not HAVE_QCEC:
        return None, "qcec_unavailable"
    try:
        res = mqt_qcec.verify(a, b)
        return _parse_qcec_result(res)
    except Exception as e:
        return None, f"qcec_error:{type(e).__name__}:{e}"

def statevector_equiv_guard(a: QuantumCircuit, b: QuantumCircuit,
                            max_qubits: int = MAX_STATEVECTOR_QUBITS,
                            seed: int = 0) -> Tuple[Optional[bool], str]:
    if a.num_qubits != b.num_qubits:
        return False, "different_num_qubits"

    n = a.num_qubits
    if n <= MAX_OPERATOR_CHECK_QUBITS:
        try:
            ok = operator_equiv_up_to_phase(a, b, max_qubits=n)
            return bool(ok), "operator_full"
        except Exception as e:
            return None, f"operator_error:{type(e).__name__}:{e}"

    if n > max_qubits:
        return None, "too_large_for_statevector_fallback"

    rnd = random.Random(seed)
    phase = None
    try:
        trials = min(16, 2**n)
        for _ in range(trials):
            bits = [rnd.randint(0, 1) for _ in range(n)]
            init = Statevector.from_label("".join(map(str, reversed(bits))))
            va = init.evolve(a).data
            vb = init.evolve(b).data
            idx = int(np.argmax(np.abs(vb)))
            if abs(vb[idx]) < 1e-12:
                if not np.allclose(va, vb, atol=1e-8):
                    return False, "statevector_counterexample"
                continue
            ph = va[idx] / vb[idx]
            if abs(ph) > 0:
                ph /= abs(ph)
            if phase is None:
                phase = ph
            if not np.allclose(va, phase * vb, atol=1e-8, rtol=0):
                return False, "statevector_counterexample"
        return None, "random_statevector_pass_not_proof"
    except Exception as e:
        return None, f"statevector_error:{type(e).__name__}:{e}"

def equivalence_guard(original: QuantumCircuit, variant: QuantumCircuit,
                      seed: int = 0) -> Tuple[str, str]:
    qv, qm = qcec_equiv(original, variant)
    if qv is True:
        return "pass", qm
    if qv is False:
        return "fail", qm

    sv, sm = statevector_equiv_guard(original, variant, seed=seed)
    if sv is True:
        return "pass", sm
    if sv is False:
        return "fail", sm
    return "unknown", f"{qm};{sm}"

# =============================================================================
# Binary symplectic Pauli model
# =============================================================================

@dataclass(frozen=True)
class PauliSig:
    x: Tuple[int, ...]
    z: Tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.x)

    def vector(self) -> np.ndarray:
        return np.array(self.x + self.z, dtype=np.uint8)

    def label(self) -> str:
        chars = []
        for xi, zi in zip(self.x, self.z):
            if xi == 0 and zi == 0:
                chars.append("I")
            elif xi == 1 and zi == 0:
                chars.append("X")
            elif xi == 0 and zi == 1:
                chars.append("Z")
            else:
                chars.append("Y")
        return "".join(chars)

    @staticmethod
    def from_vector(v: np.ndarray) -> "PauliSig":
        v = np.asarray(v, dtype=np.uint8) % 2
        n = len(v) // 2
        return PauliSig(
            tuple(int(x) for x in v[:n]),
            tuple(int(z) for z in v[n:])
        )

def symplectic_product(a: PauliSig, b: PauliSig) -> int:
    x1 = np.array(a.x, dtype=np.uint8)
    z1 = np.array(a.z, dtype=np.uint8)
    x2 = np.array(b.x, dtype=np.uint8)
    z2 = np.array(b.z, dtype=np.uint8)
    return int((int(x1 @ z2) + int(z1 @ x2)) % 2)

def pauli_z(n: int, q: int) -> PauliSig:
    x = [0] * n
    z = [0] * n
    z[q] = 1
    return PauliSig(tuple(x), tuple(z))

def gate_symplectic_matrix(n: int, name: str, qs: List[int]) -> np.ndarray:
    name = name.lower()

    def transform(v: np.ndarray) -> np.ndarray:
        v = v.copy() % 2
        x = v[:n].copy()
        z = v[n:].copy()

        if name == "h":
            q = qs[0]
            x[q], z[q] = z[q], x[q]
        elif name in ("s", "sdg"):
            q = qs[0]
            z[q] ^= x[q]
        elif name in ("x", "z", "id", "barrier"):
            pass
        elif name == "cx":
            c, t = qs
            x[t] ^= x[c]
            z[c] ^= z[t]
        else:
            raise ValueError(f"Unsupported Clifford gate in suffix: {name}")

        return np.concatenate([x, z]).astype(np.uint8)

    S = np.eye(2*n, dtype=np.uint8)
    for j in range(2*n):
        e = np.zeros(2*n, dtype=np.uint8)
        e[j] = 1
        S[:, j] = transform(e)
    return S % 2

@dataclass
class TEvent:
    event_id: int
    circuit_index: int
    qubit: int
    gate: str
    sign: int
    pauli: PauliSig

def extract_pauli_events(ct: QuantumCircuit) -> List[TEvent]:
    n = ct.num_qubits
    qmap = qid_map(ct)
    M = np.eye(2*n, dtype=np.uint8)
    rev_events: List[TEvent] = []
    data = list(ct.data)

    for idx in reversed(range(len(data))):
        inst = data[idx]
        name = inst.operation.name.lower()
        qs = [qmap[q] for q in inst.qubits]

        if name in ("t", "tdg"):
            base = pauli_z(n, qs[0]).vector()
            mapped = (M @ base) % 2
            rev_events.append(
                TEvent(
                    event_id=-1,
                    circuit_index=idx,
                    qubit=qs[0],
                    gate=name,
                    sign=(+1 if name == "t" else -1),
                    pauli=PauliSig.from_vector(mapped),
                )
            )
        elif name in ALLOWED_CT:
            if name == "barrier":
                continue
            if name in ("t", "tdg"):
                raise AssertionError("unreachable")
            Sg = gate_symplectic_matrix(n, name, qs)
            M = (M @ Sg) % 2
        else:
            raise ValueError(
                f"extract_pauli_events requires exact CT basis; found {name} at {idx}"
            )

    events = list(reversed(rev_events))
    return [
        TEvent(
            event_id=i,
            circuit_index=e.circuit_index,
            qubit=e.qubit,
            gate=e.gate,
            sign=e.sign,
            pauli=e.pauli,
        )
        for i, e in enumerate(events)
    ]

def build_anticommutation_graph(events: List[TEvent]) -> Tuple[nx.Graph, pd.DataFrame]:
    G = nx.Graph()

    for e in events:
        G.add_node(
            e.event_id,
            circuit_index=int(e.circuit_index),
            qubit=int(e.qubit),
            gate=e.gate,
            sign=int(e.sign),
            pauli=e.pauli.label(),
        )

    rows = []
    for i in range(len(events)):
        for j in range(i+1, len(events)):
            a, b = events[i], events[j]
            sp = symplectic_product(a.pauli, b.pauli)
            if sp == 1:
                G.add_edge(
                    a.event_id, b.event_id,
                    reason="pauli_anticommutation",
                    symplectic_product=1,
                )
                rows.append({
                    "u": a.event_id,
                    "v": b.event_id,
                    "u_pauli": a.pauli.label(),
                    "v_pauli": b.pauli.label(),
                    "symplectic_product": 1,
                    "reason": "pauli_anticommutation",
                })

    return G, pd.DataFrame(rows, columns=EDGE_DF_COLUMNS)


# =============================================================================
# Independent direct-matrix validation of the Pauli model
# =============================================================================

_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def pauli_matrix_qiskit_order(p: PauliSig) -> np.ndarray:
    """
    Matrix of an unsigned Pauli signature in Qiskit's little-endian qubit
    convention.  p.x/p.z are stored q0-first, while Kronecker factors for an
    n-qubit matrix are q_{n-1},...,q0.
    """
    mats = []
    for x, z in zip(reversed(p.x), reversed(p.z)):
        if x == 0 and z == 0:
            mats.append(_I2)
        elif x == 1 and z == 0:
            mats.append(_X2)
        elif x == 0 and z == 1:
            mats.append(_Z2)
        else:
            mats.append(_Y2)

    M = np.array([[1.0 + 0.0j]])
    for A in mats:
        M = np.kron(M, A)
    return M


def single_qubit_pauli_on_n(n: int, q: int, which: str = "Z") -> np.ndarray:
    if which.upper() == "Z":
        p = pauli_z(n, q)
    elif which.upper() == "X":
        x = [0] * n; z = [0] * n; x[q] = 1
        p = PauliSig(tuple(x), tuple(z))
    else:
        raise ValueError(which)
    return pauli_matrix_qiskit_order(p)


def _same_up_to_sign(A: np.ndarray, B: np.ndarray, atol: float = 1e-9) -> Tuple[bool, Optional[int]]:
    if np.allclose(A, B, atol=atol, rtol=0):
        return True, +1
    if np.allclose(A, -B, atol=atol, rtol=0):
        return True, -1
    return False, None


def _commutation_from_matrices(A: np.ndarray, B: np.ndarray, atol: float = 1e-9) -> Optional[int]:
    """
    Return 0 for commute, 1 for anticommute, None if neither relation is
    numerically certified.
    """
    AB = A @ B
    BA = B @ A
    if np.allclose(AB, BA, atol=atol, rtol=0):
        return 0
    if np.allclose(AB, -BA, atol=atol, rtol=0):
        return 1
    return None


def clifford_only_circuit(ct: QuantumCircuit,
                          start_after_index: Optional[int] = None) -> QuantumCircuit:
    """
    Preserve the chronological order of all Clifford gates while deleting
    explicit T/Tdg events.  If start_after_index is given, keep only Clifford
    gates whose circuit index is strictly larger.
    """
    out = QuantumCircuit(ct.num_qubits)
    qmap = qid_map(ct)

    for idx, inst in enumerate(ct.data):
        if start_after_index is not None and idx <= int(start_after_index):
            continue

        name = inst.operation.name.lower()
        qs = [qmap[q] for q in inst.qubits]

        if name in ("t", "tdg"):
            continue
        if name == "barrier":
            continue
        if name == "h":
            out.h(qs[0])
        elif name == "x":
            out.x(qs[0])
        elif name == "z":
            out.z(qs[0])
        elif name == "s":
            out.s(qs[0])
        elif name == "sdg":
            out.sdg(qs[0])
        elif name == "cx":
            out.cx(qs[0], qs[1])
        elif name == "id":
            out.id(qs[0])
        else:
            raise ValueError(
                f"clifford_only_circuit saw non-Clifford/unsupported gate {name}"
            )
    return out


def direct_event_axis_matrix(ct: QuantumCircuit, event: TEvent) -> np.ndarray:
    """
    Independently compute the event axis:
        P_i = C_suffix Z_q C_suffix^\\dagger,
    where C_suffix contains all Clifford gates later in circuit time than the
    explicit T/Tdg event, with all T/Tdg events omitted.

    This is independent of the binary symplectic implementation and therefore
    serves as a strong correctness cross-check.
    """
    suffix = clifford_only_circuit(ct, start_after_index=event.circuit_index)
    U = Operator(suffix).data
    Zq = single_qubit_pauli_on_n(ct.num_qubits, event.qubit, "Z")
    return U @ Zq @ U.conj().T


def validate_pauli_signatures_direct(ct: QuantumCircuit,
                                     events: List[TEvent]) -> Dict[str, Any]:
    """
    Cross-check every binary symplectic Pauli signature against direct matrix
    Clifford conjugation.  The Pauli sign is allowed to differ because the graph
    intentionally works in the projective Pauli group P/{+/-I}.
    """
    if ct.num_qubits > MAX_DIRECT_PAULI_VALIDATION_QUBITS:
        return {
            "run": False,
            "reason": f"n>{MAX_DIRECT_PAULI_VALIDATION_QUBITS}",
            "all_signature_matches": None,
            "all_pair_commutation_matches": None,
        }

    direct = {}
    sign_rows = []

    for e in events:
        Pdir = direct_event_axis_matrix(ct, e)
        Pbin = pauli_matrix_qiskit_order(e.pauli)
        ok, axis_sign = _same_up_to_sign(Pdir, Pbin)
        sign_rows.append({
            "event_id": int(e.event_id),
            "binary_pauli": e.pauli.label(),
            "axis_match_up_to_sign": bool(ok),
            "axis_sign_relative_to_binary": axis_sign,
        })
        if not ok:
            return {
                "run": True,
                "all_signature_matches": False,
                "all_pair_commutation_matches": False,
                "event_checks": sign_rows,
                "failure": f"event {e.event_id} direct axis != +/- binary axis",
            }
        direct[e.event_id] = Pdir

    pair_rows = []
    all_pairs = True
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            a, b = events[i], events[j]
            sp = symplectic_product(a.pauli, b.pauli)
            dm = _commutation_from_matrices(direct[a.event_id], direct[b.event_id])
            ok = (dm == sp)
            all_pairs = all_pairs and ok
            if SAVE_DIRECT_MATRIX_VALIDATION_DETAILS:
                pair_rows.append({
                    "u": int(a.event_id),
                    "v": int(b.event_id),
                    "symplectic_product": int(sp),
                    "direct_matrix_relation": dm,
                    "match": bool(ok),
                })

    return {
        "run": True,
        "all_signature_matches": True,
        "all_pair_commutation_matches": bool(all_pairs),
        "event_checks": sign_rows,
        "pair_checks": pair_rows if SAVE_DIRECT_MATRIX_VALIDATION_DETAILS else None,
    }


def reconstruct_pauli_rotation_unitary(ct: QuantumCircuit,
                                       events: List[TEvent]) -> Dict[str, Any]:
    """
    Independently reconstruct the CT circuit projectively as

        U ~= R(P_m,theta_m) ... R(P_1,theta_1) C_final,

    with P_i obtained by direct suffix-Clifford conjugation.

    The direct axis matrix includes the +/- Pauli sign generated by Clifford
    conjugation, so no hidden sign convention from the binary representation is
    used in this check.
    """
    if ct.num_qubits > MAX_RECONSTRUCTION_QUBITS:
        return {
            "run": False,
            "reason": f"n>{MAX_RECONSTRUCTION_QUBITS}",
            "equivalent_up_to_global_phase": None,
        }

    final_cliff = clifford_only_circuit(ct)
    Urec = Operator(final_cliff).data.astype(complex)
    dim = Urec.shape[0]
    I = np.eye(dim, dtype=complex)

    # Events are in chronological circuit order.  Left-multiplication by each
    # rotation yields R_m ... R_1 C_final after the loop.
    for e in events:
        P = direct_event_axis_matrix(ct, e)
        theta = float(e.sign) * math.pi / 4.0
        R = math.cos(theta / 2.0) * I - 1j * math.sin(theta / 2.0) * P
        Urec = R @ Urec

    Uct = Operator(ct).data
    ok = matrices_equal_up_to_global_phase(Uct, Urec, atol=MATRIX_ATOL)
    return {
        "run": True,
        "equivalent_up_to_global_phase": bool(ok),
        "qubits": int(ct.num_qubits),
        "events": int(len(events)),
    }


def deep_validate_pauli_model(ct: QuantumCircuit,
                              events: List[TEvent],
                              G: nx.Graph) -> Dict[str, Any]:
    """
    Reviewer-audit validation layer:
      A) binary signatures vs direct conjugation,
      B) symplectic relation vs direct matrix commutation,
      C) complete Pauli-rotation reconstruction vs CT unitary,
      D) graph edges exactly equal the symplectic-1 pairs.
    """
    expected_edges = {
        (events[i].event_id, events[j].event_id)
        for i in range(len(events))
        for j in range(i + 1, len(events))
        if symplectic_product(events[i].pauli, events[j].pauli) == 1
    }
    actual_edges = {tuple(sorted((int(u), int(v)))) for u, v in G.edges()}
    graph_exact = (expected_edges == actual_edges)

    sig = validate_pauli_signatures_direct(ct, events)
    recon = reconstruct_pauli_rotation_unitary(ct, events)

    return {
        "graph_edges_exactly_symplectic_one_pairs": bool(graph_exact),
        "direct_signature_validation": sig,
        "projective_rotation_reconstruction": recon,
        "pass": bool(
            graph_exact
            and (sig.get("all_signature_matches") in (True, None))
            and (sig.get("all_pair_commutation_matches") in (True, None))
            and (recon.get("equivalent_up_to_global_phase") in (True, None))
        ),
    }


def maximum_clique_witness(G: nx.Graph) -> List[int]:
    if G.number_of_nodes() == 0:
        return []
    best = max(nx.find_cliques(G), key=len, default=[])
    return sorted(int(v) for v in best)


def coloring_certificate(G: nx.Graph,
                         chi: int,
                         coloring: Dict[int, int]) -> Dict[str, Any]:
    clique = maximum_clique_witness(G)
    clique_size = len(clique)
    return {
        "chi": int(chi),
        "proper_coloring_valid": bool(is_valid_coloring(G, coloring)),
        "coloring": {str(int(v)): int(c) for v, c in coloring.items()},
        "maximum_clique_witness": clique,
        "clique_size": int(clique_size),
        "hand_checkable_exact": bool(
            is_valid_coloring(G, coloring) and clique_size == int(chi)
        ),
        "lower_bound_type": "explicit_clique" if clique_size == int(chi)
                            else "exact_search/SAT",
    }

# =============================================================================
# Exact coloring
# =============================================================================

def clique_lower_bound(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return max((len(c) for c in nx.find_cliques(G)), default=1)

def greedy_coloring_upper_bound(G: nx.Graph) -> Tuple[int, Dict[int, int]]:
    if G.number_of_nodes() == 0:
        return 0, {}
    col = nx.coloring.greedy_color(G, strategy="saturation_largest_first")
    return 1 + max(col.values(), default=-1), {int(v): int(c) for v, c in col.items()}

def is_valid_coloring(G: nx.Graph, coloring: Dict[int, int]) -> bool:
    if set(coloring) != set(G.nodes()):
        return False
    return all(coloring[u] != coloring[v] for u, v in G.edges())

def exact_k_coloring_backtrack(G: nx.Graph, k: int) -> Optional[Dict[int, int]]:
    nodes = list(G.nodes())
    if not nodes:
        return {}

    color: Dict[int, int] = {}
    neigh_colors: Dict[int, set] = {v: set() for v in nodes}

    def choose_vertex() -> int:
        uncolored = [v for v in nodes if v not in color]
        return max(uncolored, key=lambda v: (len(neigh_colors[v]), G.degree(v)))

    def rec() -> bool:
        if len(color) == len(nodes):
            return True
        v = choose_vertex()
        forbidden = {color[u] for u in G.neighbors(v) if u in color}

        for c in range(k):
            if c in forbidden:
                continue
            color[v] = c
            changed = []
            for u in G.neighbors(v):
                if u not in color and c not in neigh_colors[u]:
                    neigh_colors[u].add(c)
                    changed.append(u)
            if rec():
                return True
            for u in changed:
                neigh_colors[u].remove(c)
            del color[v]

        return False

    return dict(color) if rec() else None

def exact_chromatic_backtracking(G: nx.Graph) -> Tuple[int, Dict[int, int]]:
    if G.number_of_nodes() == 0:
        return 0, {}

    lb = max(1, clique_lower_bound(G))
    ub, _ = greedy_coloring_upper_bound(G)

    for k in range(lb, ub + 1):
        sol = exact_k_coloring_backtrack(G, k)
        if sol is not None:
            if not is_valid_coloring(G, sol):
                raise AssertionError("Invalid exact coloring")
            return k, sol

    raise RuntimeError("Exact coloring search failed")

def sat_k_colorable(G: nx.Graph, k: int) -> Tuple[bool, Dict[int, int], Dict[str, Any]]:
    """
    Exact k-colorability query with explicit encoding and solver-performance stats.

    CNF:
      - at least one color per vertex,
      - at most one color per vertex,
      - adjacent vertices cannot share a color.
    """
    if not HAVE_PYSAT:
        raise RuntimeError("python-sat unavailable")

    t0 = time.perf_counter()

    if G.number_of_nodes() == 0:
        return True, {}, {
            "k":int(k), "variables":0, "clauses":0,
            "encoding_runtime_s":0.0, "solver_runtime_s":0.0,
            "solver_stats":{}
        }

    cnf = CNF()
    var: Dict[Tuple[int, int], int] = {}

    def vv(v: int, c: int) -> int:
        key = (int(v), int(c))
        if key not in var:
            var[key] = len(var) + 1
        return var[key]

    for v in G.nodes():
        cnf.append([vv(v, c) for c in range(k)])
        for c1 in range(k):
            for c2 in range(c1+1, k):
                cnf.append([-vv(v, c1), -vv(v, c2)])

    for u, v in G.edges():
        for c in range(k):
            cnf.append([-vv(u, c), -vv(v, c)])

    t1 = time.perf_counter()
    with Minisat22(bootstrap_with=cnf.clauses) as solver:
        ts = time.perf_counter()
        ok = bool(solver.solve())
        te = time.perf_counter()
        try:
            solver_stats = dict(solver.accum_stats())
        except Exception:
            solver_stats = {}
        model_raw = solver.get_model() if ok else None

    stats = {
        "k":int(k),
        "variables":int(len(var)),
        "clauses":int(len(cnf.clauses)),
        "encoding_runtime_s":float(t1-t0),
        "solver_runtime_s":float(te-ts),
        "total_query_runtime_s":float(te-t0),
        "solver_stats":solver_stats,
    }

    if not ok:
        return False, {}, stats

    model = set(x for x in (model_raw or []) if x > 0)
    col = {}
    for v in G.nodes():
        for c in range(k):
            if vv(v, c) in model:
                col[int(v)] = int(c)
                break

    return True, col, stats

def exact_chromatic_with_crosscheck(G: nx.Graph) -> Tuple[int, Dict[int, int], Dict[str, Any]]:
    tb0 = time.perf_counter()
    chi, col = exact_chromatic_backtracking(G)
    tb1 = time.perf_counter()

    report = {
        "backtracking_chi": int(chi),
        "backtracking_valid": bool(is_valid_coloring(G, col)),
        "backtracking_runtime_s":float(tb1-tb0),
        "sat_available": bool(HAVE_PYSAT),
        "sat_crosscheck": False,
        "sat_match": None,
        "sat_calls":0,
        "sat_total_runtime_s":0.0,
        "sat_queries":[],
    }

    if USE_SAT_CROSSCHECK and HAVE_PYSAT:
        report["sat_crosscheck"] = True
        ok, scol, st = sat_k_colorable(G, chi)
        report["sat_calls"] += 1
        report["sat_total_runtime_s"] += float(st.get("total_query_runtime_s",0.0))
        report["sat_queries"].append(st)

        if not ok or not is_valid_coloring(G, scol):
            raise AssertionError("SAT disagrees with exact backtracking coloring")

        for k in range(max(1, clique_lower_bound(G)), chi):
            ok_small, _, st_small = sat_k_colorable(G, k)
            report["sat_calls"] += 1
            report["sat_total_runtime_s"] += float(st_small.get("total_query_runtime_s",0.0))
            report["sat_queries"].append(st_small)
            if ok_small:
                raise AssertionError(
                    f"SAT found coloring below exact chi: {k} < {chi}"
                )
        report["sat_match"] = True

    # Convenient headline stats for the chi-colorability encoding.
    chi_query = next(
        (q for q in report["sat_queries"] if int(q.get("k",-1)) == int(chi)),
        None
    )
    report["sat_chi_variables"] = (
        int(chi_query["variables"]) if chi_query is not None else None
    )
    report["sat_chi_clauses"] = (
        int(chi_query["clauses"]) if chi_query is not None else None
    )

    return chi, col, report

def validate_color_classes_commute(events: List[TEvent],
                                   coloring: Dict[int, int]) -> bool:
    by_id = {e.event_id: e for e in events}
    groups: Dict[int, List[int]] = {}

    for v, c in coloring.items():
        groups.setdefault(int(c), []).append(int(v))

    for members in groups.values():
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                if symplectic_product(
                    by_id[members[i]].pauli,
                    by_id[members[j]].pauli
                ) == 1:
                    return False

    return True

# =============================================================================
# Exact graph-isomorphism classification
# =============================================================================

def graph_prefilter_key(G: nx.Graph) -> Tuple:
    deg = tuple(sorted(d for _, d in G.degree()))
    return (
        G.number_of_nodes(),
        G.number_of_edges(),
        deg,
        nx.number_connected_components(G) if G.number_of_nodes() else 0,
    )

def exact_graph_classes(graphs: List[Tuple[str, nx.Graph]]) -> Dict[str, int]:
    """
    Assign exact isomorphism class IDs.
    Hash/invariants are used only as prefilters; final decision is nx.is_isomorphic.
    """
    classes: List[Tuple[Tuple, nx.Graph]] = []
    mapping: Dict[str, int] = {}

    for key, G in graphs:
        pf = graph_prefilter_key(G)
        found = None

        for cid, (cpf, rep) in enumerate(classes):
            if pf != cpf:
                continue
            if nx.is_isomorphic(G, rep):
                found = cid
                break

        if found is None:
            found = len(classes)
            classes.append((pf, G.copy()))

        mapping[key] = int(found)

    return mapping

# =============================================================================
# Normalization strategies -- V8.1 corrected
# =============================================================================

@dataclass
class NormResult:
    strategy: str
    tool: str
    success: bool
    qc: Optional[QuantumCircuit]
    trace: Dict[str, Any]
    error: Optional[str] = None


# Keep Qiskit normalization inside a declared surface basis.
# This avoids optimization levels silently emitting U2/U3/generic unitary gates,
# which made the V8 pilot impossible to rebase exactly even though the original
# circuits were generated on the pi/4 grid.
QISKIT_NORM_BASIS = [
    "h", "x", "z", "s", "sdg", "t", "tdg",
    "cx", "ccx", "rz", "rx", "ry"
]


def qiskit_basis_preserving_transpile(qc: QuantumCircuit, opt: int) -> QuantumCircuit:
    """Qiskit compiler baseline with an explicit, publication-auditable basis."""
    try:
        return transpile(
            qc,
            basis_gates=QISKIT_NORM_BASIS,
            optimization_level=int(opt),
        )
    except Exception:
        # One decomposition attempt is allowed, but still fail closed if the
        # requested basis cannot be reached.
        dec = qc.decompose(reps=10)
        return transpile(
            dec,
            basis_gates=QISKIT_NORM_BASIS,
            optimization_level=int(opt),
        )


def qiskit_strategy(name: str, opt: int):
    def run(qc: QuantumCircuit, seed: int) -> NormResult:
        try:
            out = qiskit_basis_preserving_transpile(qc, opt)
            bad = sorted(set(
                inst.operation.name.lower()
                for inst in out.data
                if inst.operation.name.lower() not in set(QISKIT_NORM_BASIS)
            ))
            if bad:
                raise RuntimeError(f"Qiskit normalization left undeclared gates: {bad}")

            return NormResult(
                strategy=name,
                tool="qiskit",
                success=True,
                qc=out,
                trace={
                    "tool":"qiskit",
                    "optimization_level":opt,
                    "basis":QISKIT_NORM_BASIS,
                    "output_gate_hist":gate_histogram(out),
                },
            )
        except Exception as e:
            return NormResult(
                strategy=name,
                tool="qiskit",
                success=False,
                qc=None,
                trace={
                    "tool":"qiskit",
                    "optimization_level":opt,
                    "basis":QISKIT_NORM_BASIS,
                    "failure_stage":"qiskit_basis_preserving_transpile",
                    "traceback":traceback.format_exc()[-3000:],
                },
                error=repr(e),
            )
    run.strategy_name = name
    return run


def circuit_to_qasm2(qc: QuantumCircuit) -> str:
    if qasm2_dumps is not None:
        return qasm2_dumps(qc)
    if hasattr(qc, "qasm"):
        return qc.qasm()
    raise RuntimeError("OpenQASM 2 export unavailable")


def sanitize_qasm2_text(txt: str) -> str:
    """
    Conservative PyZX -> Qiskit OpenQASM 2 sanitation.

    V8 pilot diagnosis showed that a large fraction of otherwise successful
    PyZX extractions failed only because PyZX emitted `swap` while the strict
    QASM2 parser did not have that symbol in scope.  Expand SWAP exactly into
    three CNOTs.  No scientific transformation is hidden here: this is only an
    exact serialization bridge.
    """
    s = str(txt or "")

    if "OPENQASM 3" in s:
        raise RuntimeError("QASM3 encountered on strict QASM2 path")

    if "OPENQASM 2" not in "\n".join(s.splitlines()[:3]):
        s = "OPENQASM 2.0;\n" + s

    if 'include "qelib1.inc";' not in s:
        lines = s.splitlines()
        inserted = False
        for i, line in enumerate(lines):
            if line.strip().startswith("OPENQASM 2"):
                lines.insert(i + 1, 'include "qelib1.inc";')
                inserted = True
                break
        if not inserted:
            lines.insert(0, 'include "qelib1.inc";')
        s = "\n".join(lines)

    def _expand_swap(m):
        a = m.group(1).strip()
        b = m.group(2).strip()
        return f"cx {a},{b};\ncx {b},{a};\ncx {a},{b};"

    s = re.sub(
        r"(?im)^\s*swap\s+([^,;]+)\s*,\s*([^;]+?)\s*;\s*$",
        _expand_swap,
        s
    )

    # Compatibility with PyZX writers that use `phase(...)`.
    s = re.sub(r"(?i)\bphase\s*\(", "u1(", s)
    return s


def circuit_from_qasm2(txt: str) -> QuantumCircuit:
    if qasm2_loads is None:
        raise RuntimeError("OpenQASM 2 loader unavailable")
    return qasm2_loads(sanitize_qasm2_text(txt))


def _graph_size(g) -> Dict[str, int]:
    return {
        "vertices": int(g.num_vertices()),
        "edges": int(g.num_edges()),
    }


def _returned_graph_or_same(ret, g):
    """PyZX API compatibility: some routines mutate, some return a graph."""
    if ret is not None and hasattr(ret, "num_vertices") and hasattr(ret, "num_edges"):
        return ret
    return g


def pyzx_common_graphlike_preprocess(g, ops: List[str]):
    """
    Apply the SAME mandatory graph-like preparation to every PyZX branch.

    This is not counted as a strategy-specific simplification.  It defines the
    common starting representation from which `no_simp`, pivot, lcomp, full
    reduce, etc. are compared.

    V8 diagnosis:
      no_simp / pivot_only / lcomp_only / teleport_reduce all failed because
      extraction requires a graph-like ZX diagram.  The corrected design makes
      graph-like conversion an explicit common preprocessing stage.
    """
    fn = getattr(zx.simplify, "to_graph_like", None)
    if fn is None:
        raise RuntimeError(
            "Installed PyZX has no zx.simplify.to_graph_like; "
            "cannot create a scientifically comparable common graph-like baseline."
        )

    before = _graph_size(g)
    ret = fn(g)
    g = _returned_graph_or_same(ret, g)
    after = _graph_size(g)

    ops.append("common_preprocess:to_graph_like")
    return g, before, after


def _apply_pyzx_strategy(g, mode: str, rnd: random.Random,
                         repeat: int, ops: List[str]):
    """Apply only the strategy-specific part AFTER common graph-like preparation."""
    for _ in range(max(1, int(repeat))):
        if mode == "none":
            ops.append("strategy:none")
            continue

        if mode == "full_reduce":
            zx.simplify.full_reduce(g)
            ops.append("strategy:full_reduce")
            continue

        if mode == "clifford_simp":
            zx.simplify.clifford_simp(g)
            ops.append("strategy:clifford_simp")
            continue

        if mode == "teleport_reduce":
            ret = zx.simplify.teleport_reduce(g)
            g = _returned_graph_or_same(ret, g)
            ops.append("strategy:teleport_reduce")
            continue

        if mode == "pivot_only":
            zx.simplify.pivot_simp(g)
            ops.append("strategy:pivot_simp")
            continue

        if mode == "lcomp_only":
            zx.simplify.lcomp_simp(g)
            ops.append("strategy:lcomp_simp")
            continue

        if mode == "random_mix":
            choices = ["pivot", "lcomp", "clifford"]
            # More than one draw is needed for an actual trajectory.
            for _j in range(4):
                r = rnd.choice(choices)
                if r == "pivot":
                    zx.simplify.pivot_simp(g)
                    ops.append("strategy_random:pivot_simp")
                elif r == "lcomp":
                    zx.simplify.lcomp_simp(g)
                    ops.append("strategy_random:lcomp_simp")
                else:
                    zx.simplify.clifford_simp(g)
                    ops.append("strategy_random:clifford_simp")
            continue

        raise ValueError(f"Unknown PyZX mode: {mode}")

    return g


def _pyzx_extract_to_qiskit(g, ops: List[str]) -> QuantumCircuit:
    """
    Exact extraction + robust QASM2 conversion.

    Direct to_qiskit methods are attempted when present; otherwise the QASM2
    route is used.  The QASM2 route expands SWAP exactly.
    """
    try:
        c2 = zx.extract.extract_circuit(g.copy())
        ops.append("extraction:extract_circuit")
    except Exception as e:
        raise RuntimeError(f"PyZX extraction failed: {type(e).__name__}: {e}") from e

    errors = []

    # API-dependent direct routes.
    try:
        basic = c2.to_basic_gates()
        if hasattr(basic, "to_qiskit"):
            out = basic.to_qiskit()
            ops.append("conversion:to_basic_gates_to_qiskit")
            return out
    except Exception as e:
        errors.append(f"basic_to_qiskit:{type(e).__name__}:{e}")

    try:
        if hasattr(c2, "to_qiskit"):
            out = c2.to_qiskit()
            ops.append("conversion:to_qiskit")
            return out
    except Exception as e:
        errors.append(f"to_qiskit:{type(e).__name__}:{e}")

    try:
        qasm = sanitize_qasm2_text(c2.to_qasm())
        out = circuit_from_qasm2(qasm)
        ops.append("conversion:qasm2_sanitized")
        return out
    except Exception as e:
        errors.append(f"qasm2_sanitized:{type(e).__name__}:{e}")

    raise RuntimeError("PyZX-to-Qiskit conversion failed: " + " | ".join(errors))


def pyzx_strategy(name: str, mode: str, repeat: int = 1,
                  fixed_random_seed: Optional[int] = None):
    """
    Corrected PyZX strategy.

    fixed_random_seed is part of the STRATEGY definition.  V8 accidentally used
    only the benchmark seed inside all random-mix functions, so strategies named
    random_mix_s0/s1/s2 could execute the same random trajectory.  V8.1 fixes
    this and records the effective RNG seed.
    """
    def run(qc: QuantumCircuit, seed: int) -> NormResult:
        if not HAVE_PYZX:
            return NormResult(
                strategy=name,
                tool="pyzx",
                success=False,
                qc=None,
                trace={"tool":"pyzx", "mode":mode, "failure_stage":"import"},
                error=f"PyZX unavailable: {globals().get('PYZX_IMPORT_ERROR')}",
            )

        trace: Dict[str, Any] = {
            "tool":"pyzx",
            "mode":mode,
            "strategy_random_seed":fixed_random_seed,
        }
        ops: List[str] = []

        try:
            # Stable circuit-specific RNG component plus fixed strategy seed.
            if fixed_random_seed is None:
                effective_seed = int(seed)
            else:
                name_digest = int(
                    hashlib.sha256(str(qc.name).encode("utf-8")).hexdigest()[:8],
                    16
                )
                effective_seed = (
                    1000003 * int(fixed_random_seed)
                    + 9176 * int(seed)
                    + name_digest
                ) & 0xFFFFFFFF

            trace["effective_random_seed"] = int(effective_seed)
            rnd = random.Random(effective_seed)

            trace["failure_stage"] = "qiskit_to_pyzx"
            c = zx.Circuit.from_qasm(circuit_to_qasm2(qc))

            trace["failure_stage"] = "circuit_to_graph"
            g = c.to_graph()
            trace["raw_graph"] = _graph_size(g)

            trace["failure_stage"] = "common_graphlike_preprocess"
            g, common_before, common_after = pyzx_common_graphlike_preprocess(g, ops)
            trace["common_graphlike_before"] = common_before
            trace["common_graphlike_after"] = common_after

            trace["strategy_input_graph"] = _graph_size(g)

            trace["failure_stage"] = "strategy_specific_rewrite"
            g = _apply_pyzx_strategy(
                g=g,
                mode=mode,
                rnd=rnd,
                repeat=repeat,
                ops=ops,
            )
            trace["strategy_output_graph"] = _graph_size(g)

            trace["failure_stage"] = "extraction_conversion"
            out = _pyzx_extract_to_qiskit(g, ops)

            # Put all PyZX outputs onto the SAME declared surface gate vocabulary
            # before equivalence/rebase.  opt=0: no additional optimization.
            trace["failure_stage"] = "post_extraction_surface_basis"
            out = qiskit_basis_preserving_transpile(out, 0)

            trace["ops"] = ops
            trace["output_gate_hist"] = gate_histogram(out)
            trace["failure_stage"] = "none"

            return NormResult(
                strategy=name,
                tool="pyzx",
                success=True,
                qc=out,
                trace=trace,
            )

        except Exception as e:
            trace["ops"] = ops
            trace["traceback"] = traceback.format_exc()[-4000:]
            return NormResult(
                strategy=name,
                tool="pyzx",
                success=False,
                qc=None,
                trace=trace,
                error=repr(e),
            )

    run.strategy_name = name
    return run


def select_tket_pass():
    if not HAVE_TKET:
        return None, None
    for name in [
        "FullPeepholeOptimise",
        "PeepholeOptimise",
        "CliffordSimp",
        "SynthesiseTket",
        "SynthesiseTK",
    ]:
        cls = getattr(_tket_passes, name, None)
        if cls is not None:
            try:
                return name, cls()
            except Exception:
                pass
    return None, None


def tket_strategy(name: str):
    def run(qc: QuantumCircuit, seed: int) -> NormResult:
        if not HAVE_TKET:
            return NormResult(
                strategy=name, tool="tket", success=False, qc=None,
                trace={"tool":"tket", "failure_stage":"import"},
                error=f"tket unavailable: {globals().get('TKET_IMPORT_ERROR')}"
            )
        try:
            tk = qiskit_to_tk(qc)
            pname, pobj = select_tket_pass()
            if pobj is not None:
                pobj.apply(tk)

            try:
                if hasattr(tk, "replace_implicit_wire_swaps"):
                    tk.replace_implicit_wire_swaps()
            except Exception:
                pass

            try:
                out = tk_to_qiskit(tk, replace_implicit_swaps=True)
            except TypeError:
                out = tk_to_qiskit(tk)

            out = qiskit_basis_preserving_transpile(out, 0)

            return NormResult(
                strategy=name,
                tool="tket",
                success=True,
                qc=out,
                trace={
                    "tool":"tket",
                    "pass":pname or "conversion_only",
                    "output_gate_hist":gate_histogram(out),
                }
            )
        except Exception as e:
            return NormResult(
                strategy=name, tool="tket", success=False, qc=None,
                trace={
                    "tool":"tket",
                    "failure_stage":"tket_normalization_or_conversion",
                    "traceback":traceback.format_exc()[-3000:]
                },
                error=repr(e)
            )
    run.strategy_name = name
    return run


def build_strategies():
    strategies = []

    if INCLUDE_QISKIT_BASELINES:
        strategies += [
            qiskit_strategy("qiskit_opt0", 0),
            qiskit_strategy("qiskit_opt1", 1),
            qiskit_strategy("qiskit_opt2", 2),
            qiskit_strategy("qiskit_opt3", 3),
        ]

    # Primary ZX analysis.
    strategies += [
        pyzx_strategy("pyzx_no_simp", "none"),
        pyzx_strategy("pyzx_full_reduce", "full_reduce"),
        pyzx_strategy("pyzx_clifford_simp", "clifford_simp"),
        pyzx_strategy("pyzx_teleport_reduce", "teleport_reduce"),
        pyzx_strategy("pyzx_pivot_only", "pivot_only"),
        pyzx_strategy("pyzx_lcomp_only", "lcomp_only"),
    ]

    # Random seeds now genuinely define distinct strategy trajectories.
    for s in PYZX_RANDOM_SEEDS:
        strategies.append(
            pyzx_strategy(
                f"pyzx_random_mix_s{s}",
                "random_mix",
                repeat=2,
                fixed_random_seed=int(s),
            )
        )

    if INCLUDE_TKET:
        strategies.append(tket_strategy("tket_compiler_norm"))

    return strategies

# =============================================================================
# Publication-scale balanced benchmark suite: 22 families x 10 circuits
# =============================================================================

def qaoa_from_edges(n: int, edges: Sequence[Tuple[int,int]], p: int,
                    name: str, gamma: float = math.pi/8,
                    beta: float = math.pi/8) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"{name}_n{n}_p{p}")
    for q in range(n): qc.h(q)
    for _ in range(int(p)):
        for i,j in [(int(a),int(b)) for a,b in edges if int(a)!=int(b)]:
            qc.cx(i,j); qc.rz(2*gamma,j); qc.cx(i,j)
        for q in range(n): qc.rx(2*beta,q)
    return qc

def qaoa_path(n,p=1): return qaoa_from_edges(n,[(i,i+1) for i in range(n-1)],p,'qaoa_path')
def qaoa_ring(n,p=1): return qaoa_from_edges(n,[(i,(i+1)%n) for i in range(n)],p,'qaoa_ring')
def qaoa_star(n,p=1): return qaoa_from_edges(n,[(0,i) for i in range(1,n)],p,'qaoa_star')
def qaoa_complete(n,p=1): return qaoa_from_edges(n,[(i,j) for i in range(n) for j in range(i+1,n)],p,'qaoa_complete')

def qaoa_erdos_renyi(n,p_edge,p,seed):
    G=nx.erdos_renyi_graph(n,p_edge,seed=seed)
    if G.number_of_edges()==0 and n>1: G.add_edges_from((i,i+1) for i in range(n-1))
    return qaoa_from_edges(n,list(G.edges()),p,f'qaoa_er_s{seed}')

def qaoa_barabasi_albert(n,m,p,seed):
    G=nx.barabasi_albert_graph(n,max(1,min(int(m),n-1)),seed=seed)
    return qaoa_from_edges(n,list(G.edges()),p,f'qaoa_ba_s{seed}')

def qaoa_random_regular(n,degree,p,seed):
    d=min(int(degree),n-1)
    if (n*d)%2: d=max(1,d-1)
    G=nx.random_regular_graph(d,n,seed=seed)
    return qaoa_from_edges(n,list(G.edges()),p,f'qaoa_rr_s{seed}')

def random_ct(n,depth,seed):
    rnd=random.Random(seed); qc=QuantumCircuit(n,name=f'random_ct_n{n}_d{depth}_s{seed}')
    for _ in range(depth):
        for q in range(n):
            r=rnd.random()
            if r<.25: qc.h(q)
            elif r<.45: qc.s(q)
            elif r<.70: qc.t(q)
            elif r<.85: qc.tdg(q)
            else: qc.rz(rnd.choice([math.pi/4,-math.pi/4,3*math.pi/4,-3*math.pi/4]),q)
        perm=list(range(n)); rnd.shuffle(perm)
        for a,b in zip(perm[::2],perm[1::2]): qc.cx(a,b)
    return qc

def random_interaction_ct(n,depth,seed,p_edge=.25):
    rnd=random.Random(seed); qc=QuantumCircuit(n,name=f'random_interaction_ct_n{n}_d{depth}_s{seed}')
    for _ in range(depth):
        for q in range(n):
            r=rnd.random()
            if r<.55: qc.rz(rnd.choice([math.pi/4,-math.pi/4,math.pi/2,-math.pi/2]),q)
            elif r<.75: qc.h(q)
            elif r<.90: qc.rx(rnd.choice([math.pi/4,-math.pi/4,math.pi/2]),q)
        for i in range(n):
            for j in range(i+1,n):
                if rnd.random()<p_edge: qc.cx(i,j)
    return qc

def random_cnot_t_layered(n,layers,seed):
    rnd=random.Random(seed); qc=QuantumCircuit(n,name=f'random_cnot_t_n{n}_l{layers}_s{seed}')
    for _ in range(layers):
        qs=list(range(n)); rnd.shuffle(qs)
        for q in qs:
            r=rnd.random()
            if r<.50: qc.t(q)
            elif r<.68: qc.tdg(q)
            elif r<.84: qc.h(q)
            else: qc.s(q)
        pairs=list(range(n)); rnd.shuffle(pairs)
        for a,b in zip(pairs[::2],pairs[1::2]): qc.cx(a,b)
        if rnd.random()<.25 and n>=3:
            a,b,c=rnd.sample(range(n),3); qc.ccx(a,b,c)
    return qc

def brickwork_ct(n,layers,seed):
    rnd=random.Random(seed); qc=QuantumCircuit(n,name=f'brickwork_n{n}_l{layers}_s{seed}')
    for l in range(layers):
        for i in range(l%2,n-1,2): qc.cx(i,i+1)
        for q in range(n):
            qc.rz(rnd.choice([math.pi/4,-math.pi/4,3*math.pi/4,-3*math.pi/4]),q)
            if rnd.random()<.35: qc.h(q)
    return qc

def dense_phase(n,rounds=1):
    qc=QuantumCircuit(n,name=f'dense_phase_n{n}_r{rounds}')
    for _ in range(rounds):
        for q in range(n): qc.h(q)
        for i in range(n):
            for j in range(i+1,n): qc.cx(i,j); qc.t(j); qc.cx(i,j)
        for q in range(n): qc.h(q)
    return qc

def qft_phase_like(n):
    qc=QuantumCircuit(n,name=f'qft_phase_like_n{n}')
    for i in range(n):
        qc.h(i)
        for j in range(i+1,n): qc.cx(j,i); qc.rz(math.pi/4,i); qc.cx(j,i)
    return qc

def controlled_phase_ladder(n,depth):
    qc=QuantumCircuit(n,name=f'controlled_phase_ladder_n{n}_d{depth}')
    for q in range(n): qc.h(q)
    for d in range(depth):
        for i in range(n-1): qc.cx(i,i+1); qc.rz(((-1)**(i+d))*math.pi/4,i+1); qc.cx(i,i+1)
        for q in (range(0,n,2) if d%2==0 else range(1,n,2)): qc.h(q)
    return qc

def phase_polynomial_stress(n,terms,seed):
    rnd=random.Random(seed); qc=QuantumCircuit(n,name=f'phase_poly_n{n}_t{terms}_s{seed}')
    for _ in range(terms):
        subset=sorted(rnd.sample(range(n),rnd.randint(2,min(n,max(2,n//2+1)))))
        pivot=subset[-1]
        for q in subset[:-1]: qc.cx(q,pivot)
        qc.rz(rnd.choice([math.pi/4,-math.pi/4,3*math.pi/4,-3*math.pi/4]),pivot)
        for q in reversed(subset[:-1]): qc.cx(q,pivot)
    return qc

def commuting_iqp(n,terms,seed):
    rnd=random.Random(seed); qc=QuantumCircuit(n,name=f'iqp_commuting_n{n}_t{terms}_s{seed}')
    for q in range(n): qc.h(q)
    for _ in range(terms):
        arity=rnd.choice([1,2,3]) if n>=3 else rnd.choice([1,2]); subset=sorted(rnd.sample(range(n),min(arity,n)))
        if len(subset)==1: qc.rz(rnd.choice([math.pi/4,-math.pi/4]),subset[0])
        else:
            pivot=subset[-1]
            for q in subset[:-1]: qc.cx(q,pivot)
            qc.rz(rnd.choice([math.pi/4,-math.pi/4]),pivot)
            for q in reversed(subset[:-1]): qc.cx(q,pivot)
    for q in range(n): qc.h(q)
    return qc

def bv_oracle_like(n,seed):
    rnd=random.Random(seed); qc=QuantumCircuit(n+1,name=f'bv_phase_like_n{n}_s{seed}'); out=n
    qc.x(out); qc.h(out)
    for q in range(n): qc.h(q)
    secret=[rnd.randint(0,1) for _ in range(n)]
    if not any(secret): secret[0]=1
    for q,b in enumerate(secret):
        if b: qc.cx(q,out); qc.rz(math.pi/4,q); qc.cx(q,out)
    for q in range(n): qc.h(q)
    return qc

def tof_chain(n):
    qc=QuantumCircuit(n,name=f'tof_chain_n{n}')
    for i in range(max(0,n-2)): qc.ccx(i,i+1,i+2)
    for i in reversed(range(max(0,n-2))): qc.ccx(i,i+1,i+2)
    return qc

def modular_adder_proxy(n):
    qc=QuantumCircuit(3*n+2,name=f'modular_adder_proxy_n{n}'); a=list(range(n)); b=list(range(n,2*n)); c=list(range(2*n,3*n+2))
    for i in range(n): qc.ccx(a[i],b[i],c[i+1]); qc.cx(a[i],b[i]); qc.ccx(c[i],b[i],c[i+1]); qc.t(b[i]); qc.tdg(c[i+1])
    for i in reversed(range(n)): qc.ccx(c[i],b[i],c[i+1]); qc.cx(a[i],b[i]); qc.ccx(a[i],b[i],c[i+1])
    return qc

def hardware_efficient_ct(n,reps,seed,entanglement='linear'):
    rnd=random.Random(seed); qc=QuantumCircuit(n,name=f'hea_{entanglement}_n{n}_r{reps}_s{seed}'); choices=[0,math.pi/4,-math.pi/4,math.pi/2,-math.pi/2]
    for _ in range(reps):
        for q in range(n): qc.rx(rnd.choice(choices),q); qc.rz(rnd.choice(choices),q)
        if entanglement=='circular':
            for q in range(n): qc.cx(q,(q+1)%n)
        elif entanglement=='full':
            for i in range(n):
                for j in range(i+1,n): qc.cx(i,j)
        else:
            for q in range(n-1): qc.cx(q,q+1)
    return qc

def real_amplitudes_ct(n,reps,seed):
    rnd=random.Random(seed); qc=QuantumCircuit(n,name=f'realamp_n{n}_r{reps}_s{seed}'); choices=[0,math.pi/4,-math.pi/4,math.pi/2,-math.pi/2]
    for _ in range(reps):
        for q in range(n): qc.ry(rnd.choice(choices),q)
        for q in range(n-1): qc.cx(q,q+1)
    for q in range(n): qc.ry(rnd.choice(choices),q)
    return qc

def vqe_su2_ct(n,reps,seed):
    rnd=random.Random(seed); qc=QuantumCircuit(n,name=f'vqe_su2_n{n}_r{reps}_s{seed}'); choices=[0,math.pi/4,-math.pi/4,math.pi/2,-math.pi/2]
    for _ in range(reps):
        for q in range(n): qc.ry(rnd.choice(choices),q); qc.rz(rnd.choice(choices),q)
        for q in range(n-1): qc.cx(q,q+1)
    return qc

def k4_motif_ct(repetitions,seed):
    rnd=random.Random(seed); qc=QuantumCircuit(4,name=f'k4_motif_r{repetitions}_s{seed}'); edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for q in range(4): qc.h(q)
    for _ in range(repetitions):
        for a,b in edges: qc.cx(a,b); qc.rz(rnd.choice([math.pi/4,-math.pi/4]),b); qc.cx(a,b)
        for q in range(4):
            if rnd.random()<.5: qc.h(q)
    return qc

@dataclass
class Benchmark:
    family: str
    circuit_id: str
    qc: QuantumCircuit
    seed: int

def make_publication_benchmarks() -> List[Benchmark]:
    """
    Predeclared balanced suite: exactly 22 families x 10 circuits = 220.

    V8.3.1 correction
    -----------------
    The first V8.3 draft could generate duplicate benchmark IDs because several
    deterministic families reused the same structural parameter tuple.  Merely
    renaming those circuits would create pseudo-replication, so V8.3.1 fixes
    the *benchmark design itself*:

      - every family has 10 predeclared, distinct parameter settings;
      - every circuit_id includes the locked suite slot;
      - exact serialized-circuit hashes are checked within each family;
      - the run aborts if any within-family duplicate circuit is detected.

    Thus "10 circuits per family" means 10 genuinely distinct generated
    circuits, not 10 labels for repeated instances.
    """
    out: List[Benchmark] = []
    suite_hashes: Dict[str, set] = {f: set() for f in [
        "BV","Path","Ring","Star","TOF","ER","ADD","RA","RICT","IQP","VQE",
        "PP","BA","RCT","RR","RCNOTT","BRICK","CPL","DENSE","QFT","K4","HEA"
    ]}

    def _local_circuit_hash(qc: QuantumCircuit) -> str:
        try:
            payload = circuit_to_qasm2(qc)
        except Exception:
            qmap = qid_map(qc)
            payload = repr([
                (
                    inst.operation.name.lower(),
                    tuple(qmap[q] for q in inst.qubits),
                    tuple(str(p) for p in getattr(inst.operation, "params", []))
                )
                for inst in qc.data
            ])
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()

    def add(family: str, qc: QuantumCircuit, seed: int, slot: int):
        if family not in suite_hashes:
            raise RuntimeError(f"Unknown publication family: {family}")
        h = _local_circuit_hash(qc)
        if h in suite_hashes[family]:
            raise RuntimeError(
                f"Duplicate generated circuit INSIDE family={family}, slot={slot}, "
                f"name={qc.name}. This is scientific pseudo-replication, so the "
                f"publication suite is aborted rather than silently relabelled."
            )
        suite_hashes[family].add(h)

        cid = safe_name(f"{family}_slot{slot:02d}_{qc.name}")
        out.append(Benchmark(
            family=family,
            circuit_id=cid,
            qc=qc,
            seed=int(seed),
        ))

    # ------------------------------------------------------------------
    # Locked structural grids
    # ------------------------------------------------------------------
    # Ten unique (n,p) combinations for deterministic QAOA topology families.
    qaoa_grid = [
        (4,1),(5,1),(6,1),(7,1),(8,1),
        (9,1),(10,1),(4,2),(6,2),(8,2),
    ]

    # Ten distinct sizes for deterministic chain/QFT families.
    tof_sizes = [4,5,6,7,8,9,10,11,12,13]
    qft_sizes = [3,4,5,6,7,8,9,10,11,12]

    # ADD remains workstation-safe by varying both width and repetition.
    # We construct repetition by composing the proxy with itself; this yields
    # genuinely distinct arithmetic-proxy circuits without inflating width to
    # impractical values.
    add_grid = [
        (2,1),(3,1),(4,1),(5,1),
        (2,2),(3,2),(4,2),
        (2,3),(3,3),(2,4),
    ]

    # Controlled-phase-ladder and dense families use unique parameter pairs.
    cpl_grid = [
        (4,1),(5,1),(6,1),(7,1),(8,1),
        (4,2),(5,2),(6,2),(7,2),(8,2),
    ]
    dense_grid = [
        (4,1),(5,1),(6,1),(7,1),(8,1),
        (4,2),(5,2),(6,2),(7,2),(8,2),
    ]

    for s in range(10):
        seed = s
        n, p = qaoa_grid[s]

        # BV -- seed changes the secret string; n also varies.
        add("BV", bv_oracle_like(max(3, n-1), seed), seed, s)

        # Deterministic topology families -- unique (n,p).
        add("Path", qaoa_path(n, p), seed, s)
        add("Ring", qaoa_ring(n, p), seed, s)
        add("Star", qaoa_star(n, p), seed, s)

        # Toffoli chain -- 10 distinct sizes.
        add("TOF", tof_chain(tof_sizes[s]), seed, s)

        # ER -- stochastic graph seed + structural grid.
        p_er = [0.20,0.25,0.30,0.35,0.40,0.22,0.28,0.33,0.38,0.45][s]
        add("ER", qaoa_erdos_renyi(n, p_er, p, seed), seed, s)

        # ADD -- width x repetition grid.
        nadd, reps_add = add_grid[s]
        base = modular_adder_proxy(nadd)
        if reps_add == 1:
            add_qc = base
        else:
            add_qc = QuantumCircuit(base.num_qubits,
                                    name=f"modular_adder_proxy_n{nadd}_r{reps_add}")
            for _ in range(reps_add):
                add_qc.compose(base, inplace=True)
        add("ADD", add_qc, seed, s)

        # Seeded variational / random families.
        add("RA", real_amplitudes_ct(n, 2 + (s % 3), seed), seed, s)
        add("RICT", random_interaction_ct(
            n,
            3 + (s % 3),
            seed,
            [0.15,0.20,0.25,0.30,0.35][s % 5]
        ), seed, s)
        add("IQP", commuting_iqp(n, 6 + 2*(s % 5), seed), seed, s)
        add("VQE", vqe_su2_ct(n, 2 + (s % 3), seed), seed, s)
        add("PP", phase_polynomial_stress(n, 6 + 2*(s % 5), seed), seed, s)
        add("BA", qaoa_barabasi_albert(n, 1 + (s % 2), p, seed), seed, s)
        add("RCT", random_ct(n, 3 + (s % 4), seed), seed, s)

        # Random regular -- seed makes same-size instances distinct.
        degree = 2 if n <= 5 else 3
        add("RR", qaoa_random_regular(n, degree, p, seed), seed, s)

        add("RCNOTT", random_cnot_t_layered(n, 5 + (s % 5), seed), seed, s)
        add("BRICK", brickwork_ct(n, 3 + (s % 5), seed), seed, s)

        # Deterministic CPL unique grid.
        ncpl, dcpl = cpl_grid[s]
        add("CPL", controlled_phase_ladder(ncpl, dcpl), seed, s)

        # Deterministic dense family unique grid.
        ndense, rdense = dense_grid[s]
        add("DENSE", dense_phase(ndense, rdense), seed, s)

        # QFT-like family -- 10 distinct sizes.
        add("QFT", qft_phase_like(qft_sizes[s]), seed, s)

        # K4 seed/repetition both influence the circuit.
        add("K4", k4_motif_ct(1 + (s % 3), seed), seed, s)

        # HEA -- entanglement topology, size, reps and seed vary.
        add("HEA", hardware_efficient_ct(
            n,
            2 + (s % 3),
            seed,
            ["linear","circular","full"][s % 3]
        ), seed, s)

    # ------------------------------------------------------------------
    # Hard scientific locks
    # ------------------------------------------------------------------
    counts = Counter(b.family for b in out)

    if LOCK_BALANCED_SUITE:
        if len(counts) != PUBLICATION_FAMILIES:
            raise RuntimeError(
                f"Balanced-suite lock failed: expected {PUBLICATION_FAMILIES} "
                f"families, got {len(counts)}: {dict(counts)}"
            )

        bad = {k:v for k,v in counts.items() if v != CIRCUITS_PER_FAMILY}
        if bad:
            raise RuntimeError(
                f"Balanced-suite lock failed: every family must contain "
                f"{CIRCUITS_PER_FAMILY} circuits; bad={bad}"
            )

        if len(out) != EXPECTED_PUBLICATION_CIRCUITS:
            raise RuntimeError(
                f"Balanced-suite lock failed: expected "
                f"{EXPECTED_PUBLICATION_CIRCUITS} circuits, got {len(out)}"
            )

    # Global IDs must be unique.
    ids = [b.circuit_id for b in out]
    if len(ids) != len(set(ids)):
        dup = [x for x,c in Counter(ids).items() if c > 1]
        raise RuntimeError(
            f"Duplicate circuit IDs in publication suite: {dup[:20]}"
        )

    # Redundant second duplicate check by family and exact serialization.
    # This is intentionally independent of circuit_id.
    for fam in sorted(counts):
        fam_b = [b for b in out if b.family == fam]
        fam_hashes = [_local_circuit_hash(b.qc) for b in fam_b]
        if len(fam_hashes) != len(set(fam_hashes)):
            raise RuntimeError(
                f"Post-build exact duplicate audit failed in family {fam}"
            )

    return out

# =============================================================================
# Core self-tests
# =============================================================================

def run_core_self_tests() -> pd.DataFrame:
    rows = []

    def record(name: str, fn):
        try:
            fn()
            rows.append({"test":name, "status":"PASS", "detail":""})
        except Exception as e:
            rows.append({
                "test":name,
                "status":"FAIL",
                "detail":f"{type(e).__name__}:{e}"
            })

    def test_xyz():
        X = PauliSig((1,), (0,))
        Y = PauliSig((1,), (1,))
        Z = PauliSig((0,), (1,))
        assert symplectic_product(X,Y) == 1
        assert symplectic_product(X,Z) == 1
        assert symplectic_product(Y,Z) == 1
        assert symplectic_product(X,X) == 0
    record("symplectic_XYZ", test_xyz)

    def test_k3():
        ev = [
            TEvent(0,0,0,"t",1,PauliSig((1,),(0,))),
            TEvent(1,1,0,"t",1,PauliSig((1,),(1,))),
            TEvent(2,2,0,"t",1,PauliSig((0,),(1,))),
        ]
        G, _ = build_anticommutation_graph(ev)
        chi, col, rep = exact_chromatic_with_crosscheck(G)
        assert chi == 3
        assert validate_color_classes_commute(ev, col)
    record("K3_chi3", test_k3)

    def test_rebase():
        qc = QuantumCircuit(3)
        qc.rx(math.pi/4, 0)
        qc.ry(-math.pi/4, 1)
        qc.rz(3*math.pi/4, 2)
        qc.ccx(0,1,2)
        ct = exact_ct_rebase(qc)
        eq = operator_equiv_up_to_phase(qc, ct, max_qubits=3)
        assert eq is True
    record("exact_rebase_operator_equiv", test_rebase)

    def test_suffix():
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.h(0)
        ct = exact_ct_rebase(qc)
        ev = extract_pauli_events(ct)
        assert ev[0].pauli == PauliSig((1,), (0,))
    record("suffix_orientation", test_suffix)

    def test_event_count():
        qc = QuantumCircuit(2)
        qc.t(0)
        qc.h(0)
        qc.tdg(1)
        qc.cx(0,1)
        qc.t(1)
        ct = exact_ct_rebase(qc)
        explicit = sum(
            inst.operation.name.lower() in ("t","tdg")
            for inst in ct.data
        )
        ev = extract_pauli_events(ct)
        assert explicit == len(ev)
    record("event_count_matches_T_count", test_event_count)


    def test_direct_signature_and_reconstruction():
        qc = QuantumCircuit(3)
        qc.t(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.tdg(1)
        qc.s(2)
        qc.cx(1, 2)
        qc.t(2)
        qc.h(2)
        ct = exact_ct_rebase(qc)
        ev = extract_pauli_events(ct)
        G, _ = build_anticommutation_graph(ev)
        deep = deep_validate_pauli_model(ct, ev, G)
        assert deep["pass"] is True
        assert deep["direct_signature_validation"]["all_signature_matches"] is True
        assert deep["direct_signature_validation"]["all_pair_commutation_matches"] is True
        assert deep["projective_rotation_reconstruction"]["equivalent_up_to_global_phase"] is True
    record("direct_signature_and_rotation_reconstruction", test_direct_signature_and_reconstruction)

    def test_zero_T_empty_graph_pipeline():
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.s(1)
        qc.cx(0,1)
        qc.h(1)
        ct = exact_ct_rebase(qc)
        ev = extract_pauli_events(ct)
        assert len(ev) == 0
        G, edge_df = build_anticommutation_graph(ev)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
        assert list(edge_df.columns) == EDGE_DF_COLUMNS
        evdf = event_dataframe(ev)
        assert list(evdf.columns) == EVENT_DF_COLUMNS
        chi, col, _ = exact_chromatic_with_crosscheck(G)
        assert chi == 0
        assert col == {}
        cert = coloring_certificate(G, chi, col)
        assert cert["chi"] == 0
        deep = deep_validate_pauli_model(ct, ev, G)
        assert deep["pass"] is True
    record("zero_T_empty_graph_pipeline", test_zero_T_empty_graph_pipeline)

    def test_hand_certificate_triangle():
        ev = [
            TEvent(0,0,0,"t",1,PauliSig((1,),(0,))),
            TEvent(1,1,0,"t",1,PauliSig((1,),(1,))),
            TEvent(2,2,0,"t",1,PauliSig((0,),(1,))),
        ]
        G, _ = build_anticommutation_graph(ev)
        chi, col, _ = exact_chromatic_with_crosscheck(G)
        cert = coloring_certificate(G, chi, col)
        assert chi == 3
        assert cert["hand_checkable_exact"] is True
        assert cert["clique_size"] == 3
    record("hand_checkable_triangle_certificate", test_hand_certificate_triangle)

    return pd.DataFrame(rows)

# =============================================================================
# Variant analysis
# =============================================================================

def circuit_serialization_hash(qc: QuantumCircuit) -> str:
    try:
        return sha256_text(circuit_to_qasm2(qc))
    except Exception:
        # Stable-ish fallback from gate sequence.
        qmap = qid_map(qc)
        parts = []
        for inst in qc.data:
            parts.append((
                inst.operation.name.lower(),
                tuple(qmap[q] for q in inst.qubits),
                tuple(str(p) for p in getattr(inst.operation, "params", []))
            ))
        return sha256_text(repr(parts))

def pauli_sequence_hash(events: List[TEvent]) -> str:
    data = [
        (e.gate, e.sign, e.pauli.label())
        for e in events
    ]
    return sha256_text(repr(data))

EVENT_DF_COLUMNS = [
    "event_id","circuit_index","qubit","gate","sign","pauli","x_bits","z_bits"
]

EDGE_DF_COLUMNS = [
    "u","v","u_pauli","v_pauli","symplectic_product","reason"
]

def event_dataframe(events: List[TEvent]) -> pd.DataFrame:
    """
    Empty-event safe event table.

    A legitimate normalization may remove every explicit T/Tdg event.  That is
    a scientifically valid zero-T representation, not an analysis error.
    Therefore the DataFrame schema is fixed even when events == [].
    """
    rows = [
        {
            "event_id": e.event_id,
            "circuit_index": e.circuit_index,
            "qubit": e.qubit,
            "gate": e.gate,
            "sign": e.sign,
            "pauli": e.pauli.label(),
            "x_bits": "".join(map(str, e.pauli.x)),
            "z_bits": "".join(map(str, e.pauli.z)),
        }
        for e in events
    ]
    return pd.DataFrame(rows, columns=EVENT_DF_COLUMNS)

def analyze_normalized_variant(
    bench: Benchmark,
    norm: NormResult,
    original: QuantumCircuit,
    strategy_dir: Path,
) -> Tuple[Dict[str, Any], Optional[nx.Graph]]:

    t_branch0 = time.perf_counter()

    row: Dict[str, Any] = {
        "family": bench.family,
        "circuit_id": bench.circuit_id,
        "seed": bench.seed,
        "strategy": norm.strategy,
        "tool": norm.tool,
        "norm_success": bool(norm.success),
        "norm_error": norm.error,
        "norm_trace_json": json.dumps(norm.trace, sort_keys=True, default=str),
        "normalization_runtime_s":float((norm.trace or {}).get("normalization_runtime_s", 0.0)),
        "original_qubits": original.num_qubits,
        "original_ops": len(original.data),
        "original_depth": int(original.depth() or 0),
    }

    if not norm.success or norm.qc is None:
        row.update({
            "equiv_status":"not_run",
            "analysis_success":False,
            "failure_category":"normalization_failed",
        })
        return row, None

    variant = norm.qc

    row["variant_qubits"] = variant.num_qubits
    row["variant_ops"] = len(variant.data)
    row["variant_depth"] = int(variant.depth() or 0)
    row["variant_gate_hist_json"] = json.dumps(
        gate_histogram(variant), sort_keys=True
    )
    row["variant_hash"] = circuit_serialization_hash(variant)

    t_eq0 = time.perf_counter()
    eq_status, eq_method = equivalence_guard(
        original, variant, seed=bench.seed
    )
    t_eq1 = time.perf_counter()
    row["equivalence_runtime_s"] = float(t_eq1-t_eq0)
    row["equiv_status"] = eq_status
    row["equiv_method"] = eq_method

    if eq_status == "fail":
        row.update({
            "analysis_success":False,
            "failure_category":"equivalence_failed",
        })
        return row, None

    t_rebase0 = time.perf_counter()
    try:
        ct = exact_ct_rebase(variant)
    except Exception as e:
        msg = repr(e)
        # Qiskit high-level compiler passes may synthesize a semantically exact
        # Clifford+T unitary into generic one-qubit rotations whose individual
        # parameters are no longer multiples of pi/4.  Such output is outside
        # the fixed event model unless an exact CT resynthesis is available.
        # Never round/approximate the angle.  Classify transparently instead.
        if "not an exact multiple of pi/4" in msg:
            category = "exact_CT_model_out_of_domain"
        else:
            category = "ct_rebase_failed"
        row.update({
            "analysis_success":False,
            "claim_eligible":False,
            "failure_category":category,
            "analysis_error":msg,
        })
        return row, None

    t_rebase1 = time.perf_counter()
    row["ct_rebase_runtime_s"] = float(t_rebase1-t_rebase0)

    row["ct_ops"] = len(ct.data)
    row["ct_depth"] = int(ct.depth() or 0)
    row["ct_t_count"] = int(sum(
        inst.operation.name.lower() in ("t","tdg")
        for inst in ct.data
    ))

    # Small-circuit independent rebase check.
    rebase_eq = operator_equiv_up_to_phase(
        variant, ct, max_qubits=MAX_OPERATOR_CHECK_QUBITS
    )
    row["rebase_equiv_checked"] = (rebase_eq is not None)
    row["rebase_equiv_pass"] = rebase_eq

    if rebase_eq is False:
        row.update({
            "analysis_success":False,
            "failure_category":"ct_rebase_equivalence_failed",
        })
        return row, None

    t_event0 = time.perf_counter()
    events = extract_pauli_events(ct)
    t_event1 = time.perf_counter()
    row["pauli_event_extraction_runtime_s"] = float(t_event1-t_event0)

    if len(events) != row["ct_t_count"]:
        raise AssertionError(
            f"{bench.circuit_id}/{norm.strategy}: "
            f"event count != explicit T-count"
        )

    t_graph0 = time.perf_counter()
    G, edge_df = build_anticommutation_graph(events)
    t_graph1 = time.perf_counter()
    row["graph_construction_runtime_s"] = float(t_graph1-t_graph0)

    t_col0 = time.perf_counter()
    chi, coloring, cross = exact_chromatic_with_crosscheck(G)
    t_col1 = time.perf_counter()
    row["exact_coloring_runtime_s"] = float(t_col1-t_col0)

    if not validate_color_classes_commute(events, coloring):
        raise AssertionError("Exact coloring has anticommuting same-color events")

    t_deep0 = time.perf_counter()
    deep_validation = (
        deep_validate_pauli_model(ct, events, G)
        if RUN_DEEP_PAULI_VALIDATION
        else {"pass": None}
    )
    t_deep1 = time.perf_counter()
    row["deep_validation_runtime_s"] = float(t_deep1-t_deep0)
    if RUN_DEEP_PAULI_VALIDATION and deep_validation.get("pass") is False:
        raise AssertionError(
            "Deep Pauli-model validation failed; branch is not publication-eligible"
        )

    cert = coloring_certificate(G, chi, coloring)

    row.update({
        "analysis_success":True,
        "claim_eligible":(eq_status == "pass"),
        "events":len(events),
        "zero_T_representation":bool(len(events) == 0),
        "empty_anticommutation_graph":bool(G.number_of_nodes() == 0),
        "pauli_sequence_hash":pauli_sequence_hash(events),
        "distinct_paulis":len({e.pauli.label() for e in events}),
        "graph_nodes":G.number_of_nodes(),
        "graph_edges":G.number_of_edges(),
        "graph_density":float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        "graph_bipartite":bool(nx.is_bipartite(G)) if G.number_of_nodes() else True,
        "graph_components":int(nx.number_connected_components(G)) if G.number_of_nodes() else 0,
        "graph_max_degree":int(max((d for _,d in G.degree()), default=0)),
        "clique_lb":int(clique_lower_bound(G)),
        "chi":int(chi),
        "color_classes_commute":True,
        "sat_crosscheck":cross.get("sat_match"),
        "backtracking_coloring_runtime_s":cross.get("backtracking_runtime_s"),
        "sat_calls":cross.get("sat_calls"),
        "sat_total_runtime_s":cross.get("sat_total_runtime_s"),
        "sat_chi_variables":cross.get("sat_chi_variables"),
        "sat_chi_clauses":cross.get("sat_chi_clauses"),
        "deep_pauli_validation_run":bool(RUN_DEEP_PAULI_VALIDATION),
        "deep_pauli_validation_pass":deep_validation.get("pass"),
        "direct_signature_validation_run":
            deep_validation.get("direct_signature_validation", {}).get("run"),
        "direct_signature_validation_pass":
            deep_validation.get("direct_signature_validation", {}).get("all_signature_matches"),
        "direct_pair_commutation_validation_pass":
            deep_validation.get("direct_signature_validation", {}).get("all_pair_commutation_matches"),
        "rotation_reconstruction_run":
            deep_validation.get("projective_rotation_reconstruction", {}).get("run"),
        "rotation_reconstruction_pass":
            deep_validation.get("projective_rotation_reconstruction", {}).get("equivalent_up_to_global_phase"),
        "full_direct_validation_completed":bool(
            deep_validation.get("direct_signature_validation", {}).get("run") is True
            and deep_validation.get("projective_rotation_reconstruction", {}).get("run") is True
        ),
        "full_direct_validation_pass":bool(
            deep_validation.get("direct_signature_validation", {}).get("all_signature_matches") is True
            and deep_validation.get("direct_signature_validation", {}).get("all_pair_commutation_matches") is True
            and deep_validation.get("projective_rotation_reconstruction", {}).get("equivalent_up_to_global_phase") is True
        ) if (
            deep_validation.get("direct_signature_validation", {}).get("run") is True
            and deep_validation.get("projective_rotation_reconstruction", {}).get("run") is True
        ) else None,
        "clique_size":int(cert["clique_size"]),
        "hand_checkable_exact_chi":bool(cert["hand_checkable_exact"]),
        "chi_lower_bound_type":cert["lower_bound_type"],
    })

    row["branch_analysis_runtime_s"] = float(time.perf_counter()-t_branch0)

    ensure_dir(strategy_dir)

    evdf = event_dataframe(events)
    # Fixed-schema event_dataframe guarantees this is safe for zero-T branches.
    evdf["color"] = evdf["event_id"].map(coloring)
    evdf.to_csv(strategy_dir / "pauli_events.csv", index=False)
    edge_df.to_csv(strategy_dir / "edge_witnesses.csv", index=False)

    json_dump(strategy_dir / "variant_report.json", {
        "row":row,
        "coloring":coloring,
        "coloring_crosscheck":cross,
        "coloring_certificate":cert,
        "deep_pauli_validation":deep_validation,
        "model_semantics":{
            "vertices":"explicit T/Tdg events after fixed exact Clifford+T rebase",
            "pauli_signature":
                "suffix-Clifford conjugated Z axis, represented modulo overall Pauli sign",
            "edge":
                "binary symplectic product 1 (Pauli anticommutation)",
            "chi":
                "minimum number of pairwise-commuting Pauli-signature classes",
            "nonclaim":
                "chi is not physical T-depth and not executable schedule depth",
        },
    })

    if SAVE_GRAPHML:
        nx.write_graphml(G, strategy_dir / "anticommutation_graph.graphml")

    if SAVE_QASM:
        try:
            (strategy_dir / "normalized.qasm").write_text(
                circuit_to_qasm2(variant), encoding="utf-8"
            )
            (strategy_dir / "rebased_ct.qasm").write_text(
                circuit_to_qasm2(ct), encoding="utf-8"
            )
        except Exception:
            pass

    return row, G

# =============================================================================
# Circuit-level aggregation
# =============================================================================

def build_circuit_summary(df: pd.DataFrame,
                          graphs: Dict[Tuple[str,str], nx.Graph]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    valid = df[
        (df["analysis_success"] == True) &
        (df["equiv_status"] == "pass")
    ].copy()

    rows = []

    for cid, g in valid.groupby("circuit_id"):
        fam = g["family"].iloc[0]

        graph_items = []
        for _, r in g.iterrows():
            key = (cid, str(r["strategy"]))
            if key in graphs:
                graph_items.append((str(r["strategy"]), graphs[key]))

        class_map = exact_graph_classes(graph_items) if graph_items else {}

        for idx, r in g.iterrows():
            strat = str(r["strategy"])
            df.loc[idx, "graph_iso_class"] = class_map.get(strat)

        chis = sorted(set(int(x) for x in g["chi"].tolist()))
        tcounts = sorted(set(int(x) for x in g["ct_t_count"].tolist()))
        iso_classes = sorted(set(int(x) for x in class_map.values()))

        # T-count-matched chromatic sensitivity.
        matched_sensitive = False
        matched_pairs = 0
        matched_graph_sensitive = False

        for tc, gt in g.groupby("ct_t_count"):
            vals = sorted(set(int(x) for x in gt["chi"].tolist()))
            if len(vals) > 1:
                matched_sensitive = True

            strats = list(gt["strategy"].astype(str))
            for a, b in itertools.combinations(strats, 2):
                matched_pairs += 1
                Ga = graphs.get((cid, a))
                Gb = graphs.get((cid, b))
                if Ga is not None and Gb is not None:
                    if not nx.is_isomorphic(Ga, Gb):
                        matched_graph_sensitive = True

        pyzx = g[g["tool"] == "pyzx"].copy()
        pyzx_chis = sorted(set(int(x) for x in pyzx["chi"].tolist())) if not pyzx.empty else []
        pyzx_graph_items = [
            (str(r["strategy"]), graphs[(cid, str(r["strategy"]))])
            for _, r in pyzx.iterrows()
            if (cid, str(r["strategy"])) in graphs
        ]
        pyzx_classes = exact_graph_classes(pyzx_graph_items) if pyzx_graph_items else {}

        rows.append({
            "circuit_id":cid,
            "family":fam,
            "valid_variants":int(len(g)),
            "valid_pyzx_variants":int(len(pyzx)),
            "distinct_T_counts":len(tcounts),
            "min_T_count":min(tcounts) if tcounts else None,
            "max_T_count":max(tcounts) if tcounts else None,
            "distinct_graph_iso_classes":len(iso_classes),
            "distinct_chi":len(chis),
            "min_chi":min(chis) if chis else None,
            "max_chi":max(chis) if chis else None,
            "chromatic_sensitive":len(chis) > 1,
            "graph_sensitive":len(iso_classes) > 1,
            "T_count_matched_chromatic_sensitive":bool(matched_sensitive),
            "T_count_matched_graph_sensitive":bool(matched_graph_sensitive),
            "T_count_matched_pairs_checked":int(matched_pairs),
            "pyzx_distinct_graph_classes":len(set(pyzx_classes.values())) if pyzx_classes else 0,
            "pyzx_distinct_chi":len(pyzx_chis),
            "pyzx_chromatic_sensitive":len(pyzx_chis) > 1,
            "pyzx_graph_sensitive":len(set(pyzx_classes.values())) > 1 if pyzx_classes else False,
        })

    return pd.DataFrame(rows)

def build_matched_pair_table(df: pd.DataFrame,
                             graphs: Dict[Tuple[str,str], nx.Graph]) -> pd.DataFrame:
    valid = df[
        (df["analysis_success"] == True) &
        (df["equiv_status"] == "pass")
    ].copy()

    rows = []

    for cid, g in valid.groupby("circuit_id"):
        fam = g["family"].iloc[0]

        for tc, gt in g.groupby("ct_t_count"):
            recs = list(gt.to_dict("records"))

            for a, b in itertools.combinations(recs, 2):
                sa = str(a["strategy"])
                sb = str(b["strategy"])
                Ga = graphs.get((cid, sa))
                Gb = graphs.get((cid, sb))
                if Ga is None or Gb is None:
                    continue

                iso = nx.is_isomorphic(Ga, Gb)

                rows.append({
                    "circuit_id":cid,
                    "family":fam,
                    "T_count":int(tc),
                    "strategy_a":sa,
                    "tool_a":a["tool"],
                    "chi_a":int(a["chi"]),
                    "edges_a":int(a["graph_edges"]),
                    "paulis_a":int(a["distinct_paulis"]),
                    "strategy_b":sb,
                    "tool_b":b["tool"],
                    "chi_b":int(b["chi"]),
                    "edges_b":int(b["graph_edges"]),
                    "paulis_b":int(b["distinct_paulis"]),
                    "graphs_isomorphic":bool(iso),
                    "graph_sensitive":not iso,
                    "chromatic_sensitive":int(a["chi"]) != int(b["chi"]),
                    "delta_chi":abs(int(a["chi"]) - int(b["chi"])),
                    "both_pyzx":(a["tool"] == "pyzx" and b["tool"] == "pyzx"),
                })

    return pd.DataFrame(rows)

def build_witness_candidates(pair_df: pd.DataFrame,
                             variant_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    cand = pair_df[
        (pair_df["chromatic_sensitive"] == True) |
        (pair_df["graph_sensitive"] == True)
    ].copy()

    if cand.empty:
        return pd.DataFrame()

    # Prefer:
    # 1) T-count-matched chromatic sensitivity
    # 2) both PyZX
    # 3) lower T-count
    # 4) lower graph size
    cand["priority_score"] = (
        cand["chromatic_sensitive"].astype(int) * 1000000 +
        cand["both_pyzx"].astype(int) * 100000 -
        cand["T_count"].astype(int) * 100 -
        (cand["edges_a"] + cand["edges_b"]).astype(int)
    )

    return cand.sort_values(
        ["priority_score", "delta_chi", "T_count"],
        ascending=[False, False, True]
    ).head(TOP_WITNESSES)

# =============================================================================
# Strategy diagnostics
# =============================================================================

def strategy_health(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for strat, g in df.groupby("strategy"):
        rows.append({
            "strategy":strat,
            "tool":g["tool"].iloc[0],
            "attempts":len(g),
            "norm_success":int(g["norm_success"].fillna(False).astype(bool).sum()),
            "equiv_pass":int((g["equiv_status"] == "pass").sum()),
            "equiv_fail":int((g["equiv_status"] == "fail").sum()),
            "equiv_unknown":int((g["equiv_status"] == "unknown").sum()),
            "analysis_success":int(g["analysis_success"].fillna(False).astype(bool).sum()),
            "claim_eligible":int(g.get("claim_eligible", pd.Series([False]*len(g))).fillna(False).astype(bool).sum()),
        })

    return pd.DataFrame(rows).sort_values(["tool","strategy"])

def distinct_representation_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[
        (df["analysis_success"] == True) &
        (df["equiv_status"] == "pass")
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    rows = []

    for cid, g in valid.groupby("circuit_id"):
        rows.append({
            "circuit_id":cid,
            "family":g["family"].iloc[0],
            "strategy_labels":int(g["strategy"].nunique()),
            "distinct_variant_hashes":int(g["variant_hash"].nunique()),
            "distinct_pauli_sequences":int(g["pauli_sequence_hash"].nunique()),
            "distinct_T_counts":int(g["ct_t_count"].nunique()),
            "distinct_chi":int(g["chi"].nunique()),
        })

    return pd.DataFrame(rows)



def failure_forensics(df: pd.DataFrame) -> pd.DataFrame:
    """Reviewer-facing exhaustive accounting of every non-success branch."""
    if df.empty:
        return pd.DataFrame()

    bad = df[
        ~(
            (df.get("analysis_success", False) == True) &
            (df.get("equiv_status", "") == "pass")
        )
    ].copy()

    if bad.empty:
        return pd.DataFrame(columns=[
            "strategy", "tool", "failure_category", "trace_failure_stage",
            "attempts", "example_error"
        ])

    rows = []
    for (strategy, tool, cat), g in bad.groupby(
        ["strategy", "tool", "failure_category"], dropna=False
    ):
        stages = Counter()
        for x in g.get("norm_trace_json", pd.Series(dtype=str)).dropna().astype(str):
            try:
                obj = json.loads(x)
                stages[str(obj.get("failure_stage", "unknown"))] += 1
            except Exception:
                stages["trace_unparseable"] += 1

        err_series = []
        for col in ["norm_error", "analysis_error"]:
            if col in g:
                err_series.extend(
                    [str(x) for x in g[col].dropna().tolist() if str(x)]
                )

        rows.append({
            "strategy":strategy,
            "tool":tool,
            "failure_category":cat,
            "trace_failure_stage_hist_json":json.dumps(dict(stages), sort_keys=True),
            "attempts":int(len(g)),
            "example_error":err_series[0][:1000] if err_series else "",
        })

    return pd.DataFrame(rows).sort_values(
        ["tool", "strategy", "failure_category"]
    )



def unique_outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prevent pseudo-replication: repeated strategy labels that produce the same
    representation/Pauli graph are not counted as independent scientific
    outcomes.
    """
    valid = df[
        (df["analysis_success"] == True) &
        (df["equiv_status"] == "pass")
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    rows = []
    for (cid, tool), g in valid.groupby(["circuit_id", "tool"]):
        rows.append({
            "circuit_id":cid,
            "family":g["family"].iloc[0],
            "tool":tool,
            "strategy_labels":int(g["strategy"].nunique()),
            "unique_variant_hashes":int(g["variant_hash"].nunique()),
            "unique_pauli_sequences":int(g["pauli_sequence_hash"].nunique()),
            "unique_T_counts":int(g["ct_t_count"].nunique()),
            "unique_chi_values":int(g["chi"].nunique()),
            "strategies_json":json.dumps(sorted(g["strategy"].astype(str).tolist())),
        })
    return pd.DataFrame(rows)


def canonical_primary_witness(pair_df: pd.DataFrame,
                              variant_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Select ONE primary witness, avoiding pair-count inflation.
    Preference:
      1) both PyZX,
      2) same T-count (pair table guarantees this),
      3) chromatic sensitivity,
      4) hand-checkable exact chi on both branches,
      5) smallest T-count / graph.
    """
    if pair_df.empty:
        return None

    cand = pair_df[
        (pair_df["both_pyzx"] == True) &
        (pair_df["chromatic_sensitive"] == True)
    ].copy()

    if cand.empty:
        return None

    lookup = variant_df.set_index(["circuit_id", "strategy"], drop=False)

    enriched = []
    for r in cand.to_dict("records"):
        try:
            a = lookup.loc[(r["circuit_id"], r["strategy_a"])]
            b = lookup.loc[(r["circuit_id"], r["strategy_b"])]
            if isinstance(a, pd.DataFrame): a = a.iloc[0]
            if isinstance(b, pd.DataFrame): b = b.iloc[0]
        except Exception:
            continue

        hand = bool(a.get("hand_checkable_exact_chi", False)) and \
               bool(b.get("hand_checkable_exact_chi", False))
        deep = bool(a.get("deep_pauli_validation_pass", False)) and \
               bool(b.get("deep_pauli_validation_pass", False))
        full_direct = bool(a.get("full_direct_validation_pass", False)) and \
                      bool(b.get("full_direct_validation_pass", False))

        rr = dict(r)
        rr["both_hand_checkable_exact"] = hand
        rr["both_deep_validated"] = deep
        rr["both_full_direct_validated"] = full_direct
        rr["size_score"] = int(r["T_count"]) + int(r["edges_a"]) + int(r["edges_b"])
        enriched.append(rr)

    if not enriched:
        return None

    edf = pd.DataFrame(enriched)
    edf = edf.sort_values(
        ["both_full_direct_validated", "both_deep_validated",
         "both_hand_checkable_exact", "size_score",
         "strategy_a", "strategy_b"],
        ascending=[False, False, False, True, True, True]
    )
    return edf.iloc[0].to_dict()


def write_primary_witness_bundle(pair_df: pd.DataFrame,
                                 variant_df: pd.DataFrame,
                                 output_dir: Path) -> Optional[Dict[str, Any]]:
    w = canonical_primary_witness(pair_df, variant_df)
    if w is None:
        return None

    cid = str(w["circuit_id"])
    sa = str(w["strategy_a"])
    sb = str(w["strategy_b"])
    wdir = output_dir / "PRIMARY_WITNESS"
    ensure_dir(wdir)

    # Copy data by re-reading the already written audit artifacts.
    source_base = output_dir / "instances" / cid
    branch_data = {}

    for label, strat in [("A", sa), ("B", sb)]:
        sdir = source_base / safe_name(strat)
        rep = json.load(open(sdir / "variant_report.json", encoding="utf-8"))
        try:
            ev = pd.read_csv(sdir / "pauli_events.csv")
        except pd.errors.EmptyDataError:
            ev = pd.DataFrame(columns=EVENT_DF_COLUMNS)
        try:
            ed = pd.read_csv(sdir / "edge_witnesses.csv")
        except pd.errors.EmptyDataError:
            ed = pd.DataFrame(columns=EDGE_DF_COLUMNS)

        ev.to_csv(wdir / f"{label}_pauli_events.csv", index=False)
        ed.to_csv(wdir / f"{label}_edge_witnesses.csv", index=False)

        for qasm_name in ["normalized.qasm", "rebased_ct.qasm"]:
            srcp = sdir / qasm_name
            if srcp.exists():
                (wdir / f"{label}_{qasm_name}").write_text(
                    srcp.read_text(encoding="utf-8", errors="ignore"),
                    encoding="utf-8"
                )

        branch_data[label] = {
            "strategy":strat,
            "row":rep.get("row", {}),
            "coloring_certificate":rep.get("coloring_certificate", {}),
            "deep_pauli_validation":rep.get("deep_pauli_validation", {}),
        }

    # Explicit lower-bound witness details.
    summary = {
        "circuit_id":cid,
        "family":w.get("family"),
        "T_count":int(w["T_count"]),
        "strategy_A":sa,
        "strategy_B":sb,
        "chi_A":int(w["chi_a"]),
        "chi_B":int(w["chi_b"]),
        "graphs_isomorphic":bool(w["graphs_isomorphic"]),
        "claim":
            "same original circuit + two semantics-preserving PyZX normalizations "
            "+ equal exact post-rebase T-count -> different Pauli anticommutation "
            "graphs and different exact chromatic numbers",
        "interpretation_guard":
            "chromatic number is a commuting-class partition number, not physical T-depth",
        "branches":branch_data,
    }

    json_dump(wdir / "witness_summary.json", summary)

    # Human-readable short certificate table.
    cert_rows = []
    for label in ["A", "B"]:
        c = branch_data[label]["coloring_certificate"]
        cert_rows.append({
            "branch":label,
            "strategy":branch_data[label]["strategy"],
            "chi":c.get("chi"),
            "clique_size":c.get("clique_size"),
            "maximum_clique_witness":json.dumps(c.get("maximum_clique_witness", [])),
            "proper_coloring_valid":c.get("proper_coloring_valid"),
            "hand_checkable_exact":c.get("hand_checkable_exact"),
            "lower_bound_type":c.get("lower_bound_type"),
        })
    pd.DataFrame(cert_rows).to_csv(
        wdir / "hand_checkable_chromatic_certificates.csv", index=False
    )
    return summary


# =============================================================================
# Reviewer-complete robustness / coverage analyses
# =============================================================================

FAMILY_DICTIONARY = {
    "BV":"Bernstein-Vazirani-like phase/oracle circuits",
    "Path":"QAOA on path graphs",
    "Ring":"QAOA on cycle/ring graphs",
    "Star":"QAOA on star graphs",
    "TOF":"Toffoli-chain circuits",
    "ER":"QAOA on Erdos-Renyi graphs",
    "ADD":"modular-adder proxy circuits",
    "RA":"RealAmplitudes-like ansatz circuits on the exact pi/4 grid",
    "RICT":"random interaction Clifford+T-grid circuits",
    "IQP":"commuting IQP-like phase circuits",
    "VQE":"EfficientSU2-like VQE ansatz circuits on the exact pi/4 grid",
    "PP":"phase-polynomial stress circuits",
    "BA":"QAOA on Barabasi-Albert graphs",
    "RCT":"random Clifford+T circuits",
    "RR":"QAOA on random-regular graphs",
    "RCNOTT":"random CNOT+T-layered circuits",
    "BRICK":"brickwork Clifford+T circuits",
    "CPL":"controlled-phase-ladder circuits",
    "DENSE":"dense pair-interaction phase circuits",
    "QFT":"QFT-like phase-interaction circuits",
    "K4":"fixed four-qubit complete-interaction motif circuits",
    "HEA":"hardware-efficient ansatz circuits",
}

def write_family_dictionary(outdir: Path) -> None:
    pd.DataFrame([
        {"family":k, "description":v}
        for k,v in sorted(FAMILY_DICTIONARY.items())
    ]).to_csv(outdir / "family_dictionary.csv", index=False)


def strategy_pair_disagreement_matrix(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Same-T-count comparisons only.  Reports whether sensitivity is broadly
    distributed across strategy pairs or concentrated in a single comparison.
    """
    if pair_df.empty:
        return pd.DataFrame()

    rows = []
    tmp = pair_df.copy()
    for _, r in tmp.iterrows():
        a,b = sorted([str(r["strategy_a"]), str(r["strategy_b"])])
        rows.append({
            "strategy_1":a,
            "strategy_2":b,
            "both_pyzx":bool(r["both_pyzx"]),
            "graph_sensitive":bool(r["graph_sensitive"]),
            "chromatic_sensitive":bool(r["chromatic_sensitive"]),
        })
    d = pd.DataFrame(rows)

    return d.groupby(
        ["strategy_1","strategy_2","both_pyzx"]
    ).agg(
        matched_T_pairs=("graph_sensitive","count"),
        graph_sensitive_pairs=("graph_sensitive","sum"),
        chromatic_sensitive_pairs=("chromatic_sensitive","sum"),
    ).reset_index().assign(
        graph_sensitive_rate=lambda x:
            x["graph_sensitive_pairs"]/x["matched_T_pairs"],
        chromatic_sensitive_rate=lambda x:
            x["chromatic_sensitive_pairs"]/x["matched_T_pairs"],
    )


def strategy_saturation_analysis(variant_df: pd.DataFrame,
                                 permutations: int = SATURATION_PERMUTATIONS) -> pd.DataFrame:
    """
    Empirical answer to Reviewer #1's 'have enough normalization strategies
    been explored?' question.

    For each circuit and each k, randomize the order of its valid PyZX
    strategies and record cumulative observed:
      - distinct exact graph-isomorphism classes,
      - distinct chi values,
      - distinct normalized-representation hashes.

    This does NOT prove exhaustive coverage.  It quantifies empirical
    saturation of the predeclared strategy set.
    """
    if variant_df.empty or "graph_iso_class" not in variant_df.columns:
        return pd.DataFrame()

    valid = variant_df[
        (variant_df["analysis_success"] == True) &
        (variant_df["equiv_status"] == "pass") &
        (variant_df["tool"] == "pyzx")
    ].copy()

    rng = random.Random(ANALYSIS_RANDOM_SEED)
    rows = []

    for cid, g in valid.groupby("circuit_id"):
        recs = g[[
            "strategy","graph_iso_class","chi","variant_hash"
        ]].drop_duplicates("strategy").to_dict("records")
        m = len(recs)
        if m == 0:
            continue

        accum = {k: {"graph":[], "chi":[], "repr":[]} for k in range(1,m+1)}

        # Deterministic base ordering plus randomized permutations.
        orders = [list(range(m))]
        for _ in range(int(permutations)):
            order = list(range(m))
            rng.shuffle(order)
            orders.append(order)

        for order in orders:
            seen_g, seen_c, seen_r = set(), set(), set()
            for kk, idxr in enumerate(order, start=1):
                rr = recs[idxr]
                seen_g.add(str(rr["graph_iso_class"]))
                seen_c.add(int(rr["chi"]))
                seen_r.add(str(rr["variant_hash"]))
                accum[kk]["graph"].append(len(seen_g))
                accum[kk]["chi"].append(len(seen_c))
                accum[kk]["repr"].append(len(seen_r))

        fam = g["family"].iloc[0]
        for k in range(1,m+1):
            for metric, key in [
                ("graph_classes","graph"),
                ("chi_values","chi"),
                ("representations","repr"),
            ]:
                vals = np.asarray(accum[k][key], dtype=float)
                rows.append({
                    "circuit_id":cid,
                    "family":fam,
                    "k_strategies":int(k),
                    "available_pyzx_strategies":int(m),
                    "metric":metric,
                    "mean":float(vals.mean()),
                    "std":float(vals.std(ddof=0)),
                    "min":float(vals.min()),
                    "max":float(vals.max()),
                    "permutations":int(len(vals)),
                })

    return pd.DataFrame(rows)


def leave_one_strategy_out_analysis(variant_df: pd.DataFrame,
                                    graphs: Dict[Tuple[str,str], nx.Graph]) -> pd.DataFrame:
    """
    Robustness check: does a circuit's same-T-count chromatic sensitivity
    disappear when one PyZX strategy is removed?
    """
    valid = variant_df[
        (variant_df["analysis_success"] == True) &
        (variant_df["equiv_status"] == "pass") &
        (variant_df["tool"] == "pyzx")
    ].copy()

    rows = []
    for cid, g in valid.groupby("circuit_id"):
        strategies = sorted(g["strategy"].astype(str).unique())
        fam = g["family"].iloc[0]

        def matched_chi_sensitive(gg: pd.DataFrame) -> bool:
            for _, gt in gg.groupby("ct_t_count"):
                if gt["chi"].nunique() > 1:
                    return True
            return False

        baseline = matched_chi_sensitive(g)

        for drop in strategies:
            gg = g[g["strategy"].astype(str) != drop]
            rows.append({
                "circuit_id":cid,
                "family":fam,
                "baseline_T_matched_chromatic_sensitive":bool(baseline),
                "dropped_strategy":drop,
                "remaining_strategies":int(gg["strategy"].nunique()),
                "sensitivity_after_drop":bool(matched_chi_sensitive(gg)),
            })

    return pd.DataFrame(rows)


def performance_summary(variant_df: pd.DataFrame) -> pd.DataFrame:
    valid = variant_df[
        (variant_df["analysis_success"] == True) &
        (variant_df["equiv_status"] == "pass")
    ].copy()
    if valid.empty:
        return pd.DataFrame()

    metrics = [
        "normalization_runtime_s",
        "equivalence_runtime_s",
        "ct_rebase_runtime_s",
        "pauli_event_extraction_runtime_s",
        "graph_construction_runtime_s",
        "exact_coloring_runtime_s",
        "backtracking_coloring_runtime_s",
        "sat_total_runtime_s",
        "deep_validation_runtime_s",
        "branch_analysis_runtime_s",
        "sat_calls",
        "sat_chi_variables",
        "sat_chi_clauses",
        "graph_nodes",
        "graph_edges",
        "ct_t_count",
    ]

    rows = []
    for metric in metrics:
        if metric not in valid.columns:
            continue
        x = pd.to_numeric(valid[metric], errors="coerce").dropna()
        if x.empty:
            continue
        rows.append({
            "metric":metric,
            "count":int(len(x)),
            "min":float(x.min()),
            "median":float(x.median()),
            "mean":float(x.mean()),
            "p95":float(x.quantile(0.95)),
            "max":float(x.max()),
        })
    return pd.DataFrame(rows)


def write_model_specification(outdir: Path) -> None:
    spec = {
        "object_chain":[
            "original circuit C",
            "normalization strategy N_s",
            "normalized/extracted circuit D_s",
            "fixed exact Clifford+T rebase R_CT(D_s)",
            "explicit T/Tdg events tau_i",
            "suffix-Clifford Pauli signatures P_i",
            "Pauli anticommutation graph G_AC",
            "exact chromatic number chi(G_AC)"
        ],
        "semantic_equivalence":
            "projective circuit equivalence; equality up to one global phase",
        "vertex":
            "one explicit T or Tdg occurrence after the fixed exact Clifford+T rebase",
        "pauli_signature":
            "P_i = C_suffix Z_q C_suffix^dagger in the projective Pauli group; "
            "implemented by binary symplectic propagation and independently "
            "cross-checked by direct matrix conjugation on small instances",
        "edge":
            "{i,j} is an edge iff P_i and P_j anticommute, equivalently their "
            "binary symplectic inner product equals 1 mod 2",
        "chromatic_interpretation":
            "minimum number of classes partitioning the event set such that "
            "all Pauli signatures within each class commute pairwise",
        "explicit_nonclaims":[
            "chi(G_AC) is not claimed to be physical T-depth",
            "chi(G_AC) is not claimed to be executable schedule depth",
            "the graph is pipeline-relative, not an intrinsic unitary invariant",
            "the finite normalization strategy set is not claimed exhaustive",
        ],
        "propositions":[
            "Pauli symplectic criterion: two Pauli signatures anticommute iff "
            "their symplectic inner product is 1",
            "Proper k-colorings of G_AC are in bijection with partitions of the "
            "event set into k pairwise-commuting classes",
            "Hence chi(G_AC) equals the minimum number of pairwise-commuting classes",
            "Global Clifford conjugation preserves the anticommutation graph",
            "Graph isomorphism implies equal chromatic number",
            "Observed chromatic diversity cannot exceed observed graph-isomorphism diversity",
            "Observed graph/chromatic diversity is monotone under enlargement of "
            "the finite strategy set",
        ],
        "controls":[
            "same-T-count pair analysis",
            "PyZX-only primary sensitivity analysis",
            "exact graph-isomorphism classification",
            "representation-hash deduplication",
            "direct Pauli-axis matrix validation",
            "projective Pauli-rotation reconstruction validation",
            "independent exact coloring and SAT cross-check",
        ],
    }
    json_dump(outdir / "model_specification.json", spec)


def write_reviewer_evidence_manifest(outdir: Path) -> None:
    manifest = {
        "Reviewer1_main_notions":{
            "status":"answered_by_code_outputs",
            "evidence":[
                "model_specification.json",
                "variant_metrics.csv",
                "instances/*/*/pauli_events.csv",
                "instances/*/*/edge_witnesses.csv",
            ]
        },
        "Reviewer1_normalization_and_extraction":{
            "status":"answered_by_code_outputs",
            "evidence":[
                "variant_metrics.csv:norm_trace_json",
                "strategy_health.csv",
                "representation_diagnostics.csv",
                "unique_outcome_summary.csv",
            ]
        },
        "Reviewer1_semantic_equivalence":{
            "status":"answered_by_code_outputs",
            "evidence":[
                "variant_metrics.csv:equiv_status,equiv_method",
                "PRIMARY_WITNESS/witness_summary.json",
            ]
        },
        "Reviewer1_full_example":{
            "status":"answered_by_primary_witness_bundle",
            "evidence":["PRIMARY_WITNESS/"]
        },
        "Reviewer1_strategy_representativeness":{
            "status":"empirically_addressed_not_proven_exhaustive",
            "evidence":[
                "strategy_saturation.csv",
                "unique_outcome_summary.csv",
                "strategy_pair_disagreement.csv",
                "leave_one_strategy_out.csv",
            ]
        },
        "Reviewer2_circular_edge_definition":{
            "status":"resolved_by_redefinition",
            "evidence":["model_specification.json","edge_witnesses.csv"]
        },
        "Reviewer2_ordering_and_Tdepth":{
            "status":"resolved_by_narrower_interpretation",
            "evidence":["model_specification.json"]
        },
        "Reviewer2_extractor_dependence":{
            "status":"explicitly_scoped_as_pipeline_relative",
            "evidence":["model_specification.json","variant_metrics.csv:norm_trace_json"]
        },
        "Reviewer2_chromatic_witness":{
            "status":"answered_by_primary_witness_bundle",
            "evidence":["PRIMARY_WITNESS/"]
        },
        "Reviewer2_verification_failures":{
            "status":"exhaustively_accounted",
            "evidence":[
                "failure_forensics.csv",
                "publication_scale_conclusion.json",
            ]
        },
        "Reviewer2_benchmark_metadata":{
            "status":"answered",
            "evidence":[
                "benchmark_manifest.csv",
                "benchmark_family_design.csv",
                "family_dictionary.csv",
            ]
        },
        "Reviewer2_runtime_scalability":{
            "status":"answered",
            "evidence":[
                "performance_summary.csv",
                "variant_metrics.csv",
            ]
        },
        "manuscript_only_items":{
            "status":"not_a_code_question",
            "items":[
                "ZX-calculus preliminaries and circuit-to-ZX exposition",
                "related-work restructuring and citations",
                "removal of duplicated manuscript sections",
                "Figure 1/Figure 5 redesign",
                "practical/theoretical discussion prose",
                "abstract/introduction/conclusion rewriting",
            ]
        }
    }
    json_dump(outdir / "reviewer_evidence_manifest.json", manifest)

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print("=" * 78)
    print("IJQI REPRODUCIBLE RESEARCH ENGINE -- PAULI ANTICOMMUTATION SENSITIVITY")
    print("=" * 78)

    environment = {
        "python_version":sys.version,
        "platform":platform.platform(),
        "numpy_version":np.__version__,
        "pandas_version":pd.__version__,
        "networkx_version":nx.__version__,
        "analysis_random_seed":ANALYSIS_RANDOM_SEED,
        "saturation_permutations":SATURATION_PERMUTATIONS,
        "processor":platform.processor(),
        "machine":platform.machine(),
        "qiskit":HAVE_QISKIT,
        "pyzx":HAVE_PYZX,
        "pysat":HAVE_PYSAT,
        "qcec":HAVE_QCEC,
        "tket":HAVE_TKET,
        "config":{
            "include_qiskit_baselines":INCLUDE_QISKIT_BASELINES,
            "include_tket":INCLUDE_TKET,
            "pyzx_random_seeds":PYZX_RANDOM_SEEDS,
            "random_seeds":RANDOM_SEEDS,
        }
    }
    json_dump(OUTPUT_DIR / "environment.json", environment)

    if not HAVE_QISKIT:
        raise RuntimeError(
            f"Qiskit is required: {globals().get('QISKIT_IMPORT_ERROR')}"
        )
    if RUN_PROFILE == "paper":
        missing = [name for name, ok in (
            ("pyzx", HAVE_PYZX), ("python-sat", HAVE_PYSAT),
            ("mqt.qcec", HAVE_QCEC)) if not ok]
        if missing:
            raise RuntimeError(
                "Full IJQI reproduction requires the verification backends: "
                + ", ".join(missing) + ". In Spyder, install them into the "
                "active interpreter. Do not count UNKNOWN equivalences as verified.")

    if RUN_CORE_SELF_TESTS:
        print("\n[1/5] Core self-tests...")
        st = run_core_self_tests()
        st.to_csv(OUTPUT_DIR / "self_test_results.csv", index=False)
        print(st.to_string(index=False))
        if not bool((st["status"] == "PASS").all()):
            raise RuntimeError(
                "Core self-test failed. Pilot aborted."
            )

    benches = make_publication_benchmarks()
    strategies = build_strategies()
    if RUN_PROFILE == "witness":
        benches = [b for b in benches if b.circuit_id == EXPECTED_WITNESS_CIRCUIT]
        strategies = [s for s in strategies if s.strategy_name in EXPECTED_WITNESS_CHI]
        if len(benches) != 1 or len(strategies) != 3:
            raise RuntimeError("Missing canonical VQE witness or its three strategies")
    elif RUN_PROFILE == "quick":
        # Different circuit families + canonical witness, without relabeling
        # these as results of the complete 220-circuit publication design.
        wanted = {"VQE", "ADD", "TOF", "RCNOTT"}
        selected = []
        seen_families = set()
        for b in benches:
            if b.family in wanted and b.family not in seen_families:
                selected.append(b)
                seen_families.add(b.family)
        selected.extend(b for b in benches if b.circuit_id == EXPECTED_WITNESS_CIRCUIT
                        and b.circuit_id not in {x.circuit_id for x in selected})
        benches = selected
        strategies = [s for s in strategies if s.strategy_name in
                      ("pyzx_no_simp", "pyzx_clifford_simp", "pyzx_pivot_only")]
    elif RUN_PROFILE != "paper":
        raise ValueError("RUN_PROFILE must be paper, quick, or witness")
    print("Run profile:", RUN_PROFILE, "| circuits:", len(benches),
          "| strategies:", len(strategies), flush=True)
    benchmark_rows = []
    for manifest_index, b in enumerate(benches, 1):
        print(f"  Manifest [{manifest_index}/{len(benches)}] {b.circuit_id}", flush=True)
        br = {
            "family":b.family,
            "circuit_id":b.circuit_id,
            "seed":b.seed,
            "qubits":b.qc.num_qubits,
            "ops":len(b.qc.data),
            "depth":int(b.qc.depth() or 0),
            "gate_hist_json":json.dumps(gate_histogram(b.qc),sort_keys=True),
        }
        try:
            bct = exact_ct_rebase(b.qc)
            br["original_CT_rebase_success"] = True
            br["original_CT_ops"] = int(len(bct.data))
            br["original_CT_depth"] = int(bct.depth() or 0)
            br["original_CT_T_count"] = int(sum(
                inst.operation.name.lower() in ("t","tdg")
                for inst in bct.data
            ))
        except Exception as e:
            br["original_CT_rebase_success"] = False
            br["original_CT_error"] = repr(e)
        benchmark_rows.append(br)
    benchmark_manifest = pd.DataFrame(benchmark_rows)
    benchmark_manifest.to_csv(OUTPUT_DIR / "benchmark_manifest.csv", index=False)
    family_design = benchmark_manifest.groupby("family").agg(
        circuits=("circuit_id","count"), min_qubits=("qubits","min"),
        max_qubits=("qubits","max"), min_ops=("ops","min"),
        max_ops=("ops","max"), min_depth=("depth","min"),
        max_depth=("depth","max")
    ).reset_index()
    family_design.to_csv(OUTPUT_DIR / "benchmark_family_design.csv", index=False)

    print(f"\n[2/5] Benchmarks: {len(benches)} across {benchmark_manifest['family'].nunique()} balanced families")
    print(f"[3/5] Strategies: {len(strategies)}")
    print("       " + ", ".join(
        getattr(s, "__name__", "strategy") for s in strategies[:3]
    ) + (" ..." if len(strategies) > 3 else ""))

    rows: List[Dict[str, Any]] = []
    graphs: Dict[Tuple[str,str], nx.Graph] = {}
    completed = set()
    source_fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    integrity_file = OUTPUT_DIR / "run_integrity.json"
    if RESUME_INTERRUPTED_RUN and (OUTPUT_DIR / "variant_metrics_partial.csv").exists():
        if not integrity_file.exists():
            raise RuntimeError("Existing partial run has no integrity fingerprint. "
                               "Move the output directory aside or disable resume.")
        previous = json.loads(integrity_file.read_text(encoding="utf-8"))
        if (previous.get("source_sha256") != source_fingerprint or
                previous.get("profile") != RUN_PROFILE or
                previous.get("strategies") != [s.strategy_name for s in strategies]):
            raise RuntimeError("Partial output was produced with different source/config. "
                               "A changed scientific pipeline cannot silently resume it.")
        cached = pd.read_csv(OUTPUT_DIR / "variant_metrics_partial.csv")
        for item in cached.to_dict("records"):
            cid, st = str(item.get("circuit_id", "")), str(item.get("strategy", ""))
            if cid not in {b.circuit_id for b in benches} or st not in {s.strategy_name for s in strategies}:
                continue
            if item.get("analysis_success") is True and str(item.get("equiv_status")) == "pass":
                gp = OUTPUT_DIR / "instances" / cid / st / "anticommutation_graph.graphml"
                if not gp.is_file():
                    print("  Resume: missing graph, recomputing", cid, st)
                    continue
                try:
                    recovered = nx.read_graphml(gp)
                    recovered = nx.relabel_nodes(recovered, {v:int(v) for v in recovered.nodes()})
                    graphs[(cid, st)] = recovered
                except Exception as err:
                    print("  Resume: invalid GraphML, recomputing", cid, st, err)
                    continue
            rows.append(item)
            completed.add((cid,st))
        print(f"  Resume: {len(completed)} previously completed branches reused.", flush=True)
    else:
        json_dump(integrity_file, {
            "source_sha256":source_fingerprint,
            "profile":RUN_PROFILE,
            "strategies":[s.strategy_name for s in strategies],
        })

    for i, bench in enumerate(benches, start=1):
        print(
            f"\n[{i}/{len(benches)}] "
            f"{bench.circuit_id}  family={bench.family}"
        )

        original = bench.qc

        cdir = OUTPUT_DIR / "instances" / bench.circuit_id
        ensure_dir(cdir)

        if SAVE_QASM:
            try:
                (cdir / "original.qasm").write_text(
                    circuit_to_qasm2(original),
                    encoding="utf-8"
                )
            except Exception:
                pass

        json_dump(cdir / "original_meta.json", {
            "family":bench.family,
            "circuit_id":bench.circuit_id,
            "qubits":original.num_qubits,
            "ops":len(original.data),
            "depth":int(original.depth() or 0),
            "gate_hist":gate_histogram(original),
        })

        for strat_fn in strategies:
            if (bench.circuit_id, strat_fn.strategy_name) in completed:
                continue
            # Recover strategy name by one dry attribute-free convention:
            # each function returns NormResult containing its own name.
            t_norm0 = time.perf_counter()
            try:
                norm = strat_fn(original, bench.seed)
            except Exception as e:
                norm = NormResult(
                    strategy=getattr(strat_fn, "__name__", "unknown_strategy"),
                    tool="unknown",
                    success=False,
                    qc=None,
                    trace={"traceback":traceback.format_exc()[-3000:]},
                    error=repr(e),
                )
            t_norm1 = time.perf_counter()
            norm.trace = dict(norm.trace or {})
            norm.trace["normalization_runtime_s"] = float(t_norm1-t_norm0)

            sdir = cdir / safe_name(norm.strategy)

            try:
                row, G = analyze_normalized_variant(
                    bench,
                    norm,
                    original,
                    sdir,
                )
            except Exception as e:
                row = {
                    "family":bench.family,
                    "circuit_id":bench.circuit_id,
                    "seed":bench.seed,
                    "strategy":norm.strategy,
                    "tool":norm.tool,
                    "norm_success":bool(norm.success),
                    "equiv_status":"not_run",
                    "analysis_success":False,
                    "failure_category":"analysis_exception",
                    "analysis_error":repr(e),
                    "traceback":traceback.format_exc()[-3000:],
                }
                G = None

            rows.append(row)

            if G is not None:
                graphs[(bench.circuit_id, norm.strategy)] = G

            print(
                f"  {norm.strategy:28s} "
                f"norm={row.get('norm_success')} "
                f"eq={row.get('equiv_status')} "
                f"T={row.get('ct_t_count')} "
                f"E={row.get('graph_edges')} "
                f"chi={row.get('chi')}"
            )

        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / "variant_metrics_partial.csv",
            index=False
        )

    print("\n[4/5] Aggregating exact graph classes and matched-T comparisons...")

    variant_df = pd.DataFrame(rows)

    # First circuit summary assigns exact graph class IDs back into variant_df.
    circuit_summary = build_circuit_summary(
        variant_df,
        graphs
    )

    variant_df.to_csv(
        OUTPUT_DIR / "variant_metrics.csv",
        index=False
    )
    circuit_summary.to_csv(
        OUTPUT_DIR / "circuit_summary.csv",
        index=False
    )
    if not circuit_summary.empty:
        family_summary = circuit_summary.groupby("family").agg(
            circuits=("circuit_id","count"),
            graph_sensitive=("graph_sensitive","sum"),
            chromatic_sensitive=("chromatic_sensitive","sum"),
            T_matched_graph_sensitive=("T_count_matched_graph_sensitive","sum"),
            T_matched_chromatic_sensitive=("T_count_matched_chromatic_sensitive","sum"),
            pyzx_graph_sensitive=("pyzx_graph_sensitive","sum"),
            pyzx_chromatic_sensitive=("pyzx_chromatic_sensitive","sum")
        ).reset_index()
        for c in ["graph_sensitive","chromatic_sensitive","T_matched_graph_sensitive","T_matched_chromatic_sensitive","pyzx_graph_sensitive","pyzx_chromatic_sensitive"]:
            family_summary[c+"_rate"] = family_summary[c] / family_summary["circuits"]
        family_summary.to_csv(OUTPUT_DIR / "family_sensitivity_summary.csv", index=False)

    matched_pairs = build_matched_pair_table(
        variant_df,
        graphs
    )
    matched_pairs.to_csv(
        OUTPUT_DIR / "T_count_matched_pairs.csv",
        index=False
    )

    witnesses = build_witness_candidates(
        matched_pairs,
        variant_df
    )
    witnesses.to_csv(
        OUTPUT_DIR / "witness_candidates.csv",
        index=False
    )

    health = strategy_health(variant_df)
    health.to_csv(
        OUTPUT_DIR / "strategy_health.csv",
        index=False
    )

    repdiag = distinct_representation_diagnostics(variant_df)
    repdiag.to_csv(
        OUTPUT_DIR / "representation_diagnostics.csv",
        index=False
    )

    forensic = failure_forensics(variant_df)
    forensic.to_csv(
        OUTPUT_DIR / "failure_forensics.csv",
        index=False
    )

    unique_outcomes = unique_outcome_summary(variant_df)
    unique_outcomes.to_csv(
        OUTPUT_DIR / "unique_outcome_summary.csv",
        index=False
    )

    primary_witness = write_primary_witness_bundle(
        matched_pairs,
        variant_df,
        OUTPUT_DIR
    )

    # Reviewer-complete robustness / coverage evidence.
    write_family_dictionary(OUTPUT_DIR)
    write_model_specification(OUTPUT_DIR)
    write_reviewer_evidence_manifest(OUTPUT_DIR)

    strat_pair = strategy_pair_disagreement_matrix(matched_pairs)
    strat_pair.to_csv(
        OUTPUT_DIR / "strategy_pair_disagreement.csv", index=False
    )

    if RUN_STRATEGY_SATURATION:
        saturation = strategy_saturation_analysis(
            variant_df, permutations=SATURATION_PERMUTATIONS
        )
    else:
        saturation = pd.DataFrame()
    saturation.to_csv(
        OUTPUT_DIR / "strategy_saturation.csv", index=False
    )

    if RUN_LEAVE_ONE_STRATEGY_OUT:
        loo = leave_one_strategy_out_analysis(variant_df, graphs)
    else:
        loo = pd.DataFrame()
    loo.to_csv(
        OUTPUT_DIR / "leave_one_strategy_out.csv", index=False
    )

    perf = performance_summary(variant_df)
    perf.to_csv(
        OUTPUT_DIR / "performance_summary.csv", index=False
    )

    print("\n[5/5] Publication-scale conclusion...")

    valid = variant_df[
        (variant_df["analysis_success"] == True) &
        (variant_df["equiv_status"] == "pass")
    ].copy()

    result = {
        "benchmarks":int(len(benches)),
        "families":int(len(set(b.family for b in benches))),
        "circuits_per_family_target":int(CIRCUITS_PER_FAMILY) if RUN_PROFILE == "paper" else None,
        "balanced_suite_locked":bool(LOCK_BALANCED_SUITE and RUN_PROFILE == "paper"),
        "strategies":int(len(strategies)),
        "planned_strategy_level_branches":int(len(benches)*len(strategies)),
        "valid_equivalent_variants":int(len(valid)),
        "circuits_summarized":int(len(circuit_summary)),
        "graph_sensitive_circuits":int(
            circuit_summary["graph_sensitive"].sum()
        ) if not circuit_summary.empty else 0,
        "chromatic_sensitive_circuits":int(
            circuit_summary["chromatic_sensitive"].sum()
        ) if not circuit_summary.empty else 0,
        "T_count_matched_graph_sensitive_circuits":int(
            circuit_summary["T_count_matched_graph_sensitive"].sum()
        ) if not circuit_summary.empty else 0,
        "T_count_matched_chromatic_sensitive_circuits":int(
            circuit_summary["T_count_matched_chromatic_sensitive"].sum()
        ) if not circuit_summary.empty else 0,
        "pyzx_graph_sensitive_circuits":int(
            circuit_summary["pyzx_graph_sensitive"].sum()
        ) if not circuit_summary.empty else 0,
        "pyzx_chromatic_sensitive_circuits":int(
            circuit_summary["pyzx_chromatic_sensitive"].sum()
        ) if not circuit_summary.empty else 0,
        "max_chi":int(valid["chi"].max()) if not valid.empty else None,
        "max_graph_edges":int(valid["graph_edges"].max()) if not valid.empty else None,
        "max_T_count":int(valid["ct_t_count"].max()) if not valid.empty else None,
        "matched_pair_rows":int(len(matched_pairs)),
        "normalization_failures":int(
            (variant_df.get("failure_category", "") == "normalization_failed").sum()
        ) if not variant_df.empty else 0,
        "ct_rebase_failures":int(
            (variant_df.get("failure_category", "") == "ct_rebase_failed").sum()
        ) if not variant_df.empty else 0,
        "exact_CT_model_out_of_domain":int(
            (variant_df.get("failure_category", "") == "exact_CT_model_out_of_domain").sum()
        ) if not variant_df.empty else 0,
        "zero_T_valid_variants":int(
            (
                (variant_df.get("analysis_success", False) == True) &
                (variant_df.get("equiv_status", "") == "pass") &
                (variant_df.get("zero_T_representation", False) == True)
            ).sum()
        ) if not variant_df.empty else 0,
        "equivalence_failures":int(
            (variant_df.get("equiv_status", "") == "fail").sum()
        ) if not variant_df.empty else 0,
        "equivalence_unknown":int(
            (variant_df.get("equiv_status", "") == "unknown").sum()
        ) if not variant_df.empty else 0,
        "matched_chromatic_sensitive_pairs":int(
            matched_pairs["chromatic_sensitive"].sum()
        ) if not matched_pairs.empty else 0,
        "matched_graph_sensitive_pairs":int(
            matched_pairs["graph_sensitive"].sum()
        ) if not matched_pairs.empty else 0,
        "both_pyzx_matched_chromatic_sensitive_pairs":int(
            (
                (matched_pairs["both_pyzx"] == True) &
                (matched_pairs["chromatic_sensitive"] == True)
            ).sum()
        ) if not matched_pairs.empty else 0,
        "deep_pauli_validated_variants":int(
            (valid.get("deep_pauli_validation_pass", False) == True).sum()
        ) if not valid.empty else 0,
        "direct_signature_validated_variants":int(
            (valid.get("direct_signature_validation_pass", False) == True).sum()
        ) if not valid.empty else 0,
        "rotation_reconstruction_validated_variants":int(
            (valid.get("rotation_reconstruction_pass", False) == True).sum()
        ) if not valid.empty else 0,
        "hand_checkable_exact_chi_variants":int(
            (valid.get("hand_checkable_exact_chi", False) == True).sum()
        ) if not valid.empty else 0,
        "full_direct_validation_completed_variants":int(
            (valid.get("full_direct_validation_completed", False) == True).sum()
        ) if not valid.empty else 0,
        "full_direct_validation_pass_variants":int(
            (valid.get("full_direct_validation_pass", False) == True).sum()
        ) if not valid.empty else 0,
        "primary_witness_available":bool(primary_witness is not None),
        "primary_witness_circuit":
            primary_witness.get("circuit_id") if primary_witness else None,
        "primary_witness_strategies":
            [primary_witness.get("strategy_A"), primary_witness.get("strategy_B")]
            if primary_witness else None,
        "strategy_saturation_rows":int(len(saturation)),
        "leave_one_strategy_out_rows":int(len(loo)),
        "strategy_pair_disagreement_rows":int(len(strat_pair)),
        "performance_summary_rows":int(len(perf)),
        "original_manuscript_reference":{
            "circuits":201,"families":22,"strategy_level_executions":2814,
            "verified_representations":2789,"fully_stable":3,
            "graph_sensitive_chromatically_stable":145,"chromatically_sensitive":53,
            "note":"Historical counts used the superseded wire-local model and are not targets."
        },
        "design_exceeds_original_scale":bool(len(benches)>201 and len(benches)*len(strategies)>2814),
        "interpretation_guard":
            "chi is the exact chromatic number of the Pauli anticommutation graph, "
            "interpreted as minimum mutually commuting Pauli-signature classes; "
            "it is not physical T-depth or executable scheduling depth.",
    }

    # GO / CAUTION / NO-GO logic for the next stage.
    if (
        result["T_count_matched_chromatic_sensitive_circuits"] > 0
        and result["primary_witness_available"]
    ):
        result["pilot_decision"] = "GO_STRONG"
        result["pilot_reason"] = (
            "At least one exact T-count-matched PyZX chromatic-sensitivity witness "
            "exists and has been exported as a distinct primary witness bundle."
        )
    elif result["T_count_matched_graph_sensitive_circuits"] > 0:
        result["pilot_decision"] = "GO_GRAPH_ONLY_CAUTION"
        result["pilot_reason"] = (
            "T-count-matched graph sensitivity exists, but no matched chromatic "
            "sensitivity was found in this pilot."
        )
    elif result["graph_sensitive_circuits"] > 0:
        result["pilot_decision"] = "CAUTION_CONFOUNDED"
        result["pilot_reason"] = (
            "Graph sensitivity exists only without a T-count-matched witness."
        )
    else:
        result["pilot_decision"] = "NO_GO_CURRENT_CLAIM"
        result["pilot_reason"] = (
            "The revised abstraction showed no representation sensitivity in the pilot."
        )

    json_dump(
        OUTPUT_DIR / "publication_scale_conclusion.json",
        result
    )

    reviewer_summary = f"""V8.4 REVIEWER-COMPLETE RUN SUMMARY
==================================
Model:
  vertices = explicit T/Tdg events after fixed exact Clifford+T rebase
  edge     = Pauli anticommutation (binary symplectic product = 1)
  chi      = minimum number of pairwise-commuting Pauli-signature classes
  nonclaim = not physical T-depth; not executable schedule depth

Coverage:
  circuits: {result.get('benchmarks')}
  families: {result.get('families')}
  strategies: {result.get('strategies')}
  planned branches: {result.get('planned_strategy_level_branches')}
  valid equivalent analyzed: {result.get('valid_equivalent_variants')}

Sensitivity:
  graph-sensitive circuits: {result.get('graph_sensitive_circuits')}
  chromatically-sensitive circuits: {result.get('chromatic_sensitive_circuits')}
  T-count-matched graph-sensitive circuits: {result.get('T_count_matched_graph_sensitive_circuits')}
  T-count-matched chromatically-sensitive circuits: {result.get('T_count_matched_chromatic_sensitive_circuits')}
  PyZX graph-sensitive circuits: {result.get('pyzx_graph_sensitive_circuits')}
  PyZX chromatically-sensitive circuits: {result.get('pyzx_chromatic_sensitive_circuits')}

Validation:
  equivalence failures: {result.get('equivalence_failures')}
  equivalence unknown: {result.get('equivalence_unknown')}
  exact-CT out-of-domain: {result.get('exact_CT_model_out_of_domain')}
  direct validation completed: {result.get('full_direct_validation_completed_variants')}
  direct validation passed: {result.get('full_direct_validation_pass_variants')}
  hand-checkable exact chi variants: {result.get('hand_checkable_exact_chi_variants')}
  primary witness: {result.get('primary_witness_circuit')}

Important:
  Finite strategy saturation is empirical evidence only, not a proof of exhaustive
  normalization-space coverage.  The graph and chi are explicitly pipeline-relative.
"""
    (OUTPUT_DIR / "reviewer_complete_run_summary.txt").write_text(
        reviewer_summary, encoding="utf-8"
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nOutputs: {OUTPUT_DIR.resolve()}")

    if result["pilot_decision"] == "GO_STRONG":
        print("\nRESULT: GO_STRONG")
        print("The next step is a reviewer-complete full run plus explicit witness figures.")
    elif result["pilot_decision"] == "GO_GRAPH_ONLY_CAUTION":
        print("\nRESULT: GO_GRAPH_ONLY_CAUTION")
        print("The original structural-sensitivity claim survives, but the chromatic claim "
              "needs more evidence or narrower wording.")
    elif result["pilot_decision"] == "CAUTION_CONFOUNDED":
        print("\nRESULT: CAUTION_CONFOUNDED")
        print("Do not launch the full publication run yet.")
    else:
        print("\nRESULT: NO_GO_CURRENT_CLAIM")
        print("Do not launch the full publication run.")




# ========================================================================
# SINGLE-FILE IJQI PUBLICATION EVIDENCE AND EXACT CHANNEL CERTIFICATION
# ========================================================================
from fractions import Fraction as F
import csv
import zipfile
from datetime import datetime, timezone

# The following independent verifier operates over Q(sqrt(2)) using Fractions.
# Its exact Pauli-transfer coefficients never use floating-point comparison.
ZERO = (F(0), F(0))
ONE = (F(1), F(0))

def add(a, b):
    return a[0] + b[0], a[1] + b[1]

def mulsign(a, s):
    return a[0] * s, a[1] * s

def divsqrt2(a):
    return a[1], a[0]/2

def sde(a):
    # Least k such that (sqrt(2))^k * a belongs to Z[sqrt(2)].
    for k in range(100):
        if k % 2 == 0:
            v = (a[0] * 2**(k//2), a[1] * 2**(k//2))
        else:
            v = (a[1]*2**((k+1)//2), a[0]*2**(k//2))
        if v[0].denominator == 1 and v[1].denominator == 1:
            return k
    raise ArithmeticError('SDE unexpectedly large')

SINGLE = {
 'h': {'I':('I',1),'X':('Z',1),'Y':('Y',-1),'Z':('X',1)},
 's': {'I':('I',1),'X':('Y',1),'Y':('X',-1),'Z':('Z',1)},
 'sdg': {'I':('I',1),'X':('Y',-1),'Y':('X',1),'Z':('Z',1)},
 'x': {'I':('I',1),'X':('X',1),'Y':('Y',-1),'Z':('Z',-1)},
 'z': {'I':('I',1),'X':('X',-1),'Y':('Y',-1),'Z':('Z',1)},
}
P1 = {'I':np.eye(2), 'X':np.array([[0,1],[1,0]]), 'Y':np.array([[0,-1j],[1j,0]]), 'Z':np.diag([1,-1])}
CX = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]],dtype=complex)
CXMAP = {}
for a in 'IXYZ':
 for b in 'IXYZ':
  M = CX @ np.kron(P1[a],P1[b]) @ CX.conj().T
  sols = [(x,y,sign) for x in 'IXYZ' for y in 'IXYZ' for sign in [1,-1]
          if np.allclose(M,sign*np.kron(P1[x],P1[y]),atol=1e-13)]
  assert len(sols)==1, (a,b,sols)
  CXMAP[a,b] = sols[0]

INST = re.compile(r'^\s*(h|x|z|s|sdg|t|tdg|cx|id)\s+q\[(\d+)\](?:\s*,\s*q\[(\d+)\])?\s*;\s*$', re.I)
QREG = re.compile(r'^\s*qreg\s+q\[(\d+)\];',re.I)

def parse_qasm(qasm):
 n=None;ops=[]
 for line in qasm.splitlines():
  line=line.split('//',1)[0].strip()
  if not line:continue
  qreg=QREG.match(line)
  if qreg:n=int(qreg.group(1));continue
  if line.startswith(('OPENQASM','include','barrier')):continue
  m=INST.match(line)
  if not m: raise ValueError('Unsupported QASM line: '+line)
  gate=m.group(1).lower();qs=tuple(int(v) for v in m.groups()[1:] if v is not None)
  if n is None:raise ValueError('Qreg must come before gates')
  ops.append((gate,qs))
 assert n is not None
 return n,ops

def transform(d,gate,qs):
 out={}
 def accum(st,c):
  out[st]=add(out.get(st,ZERO),c)
 for st,c in d.items():
  q=list(st)
  if gate in SINGLE:
   v,sign=SINGLE[gate][q[qs[0]]]
   q[qs[0]]=v
   accum(''.join(q),mulsign(c,sign))
  elif gate in ('t','tdg'):
   idx=qs[0]
   if q[idx] in ('I','Z'):accum(st,c);continue
   sgn=1 if gate=='t' else -1
   accum(st,divsqrt2(c))
   q[idx]='Y' if q[idx]=='X' else 'X'
   # T: X -> +Y, Y -> -X ; inverse signs for Tdg.
   new_sign=sgn*(1 if st[idx]=='X' else -1)
   accum(''.join(q),mulsign(divsqrt2(c),new_sign))
  elif gate=='cx':
   ct,tg=qs
   x,y,sign=CXMAP[q[ct],q[tg]]
   q[ct]=x;q[tg]=y
   accum(''.join(q),mulsign(c,sign))
  elif gate=='id':accum(st,c)
  else: raise ValueError(gate)
 return {p:c for p,c in out.items() if c!=ZERO}

def exact_channel(qasm):
 n,ops=parse_qasm(qasm)
 result={}
 for inp in map(''.join,itertools.product('IXYZ',repeat=n)):
  d={inp:ONE}
  for g,qs in ops:d=transform(d,g,qs)
  for oup,c in d.items():result[(inp,oup)]=c
 return result

def verify(qasm):
 n,ops=parse_qasm(qasm)
 if n>6:raise ValueError('Restricted to <=6 qubits for exact enumeration')
 t_count=sum(g in ('t','tdg') for g,qs in ops)
 max_sde=0;witness=None;stats=Counter()
 for inp in map(''.join,itertools.product('IXYZ',repeat=n)):
  d={inp:ONE}
  for g,qs in ops:d=transform(d,g,qs)
  for oup,c in d.items():
   v=sde(c)
   stats[v]+=1
   if v>max_sde:
    max_sde=v;witness=(inp,oup,str(c[0]),str(c[1]))
 return {'qubits':n,'T_count':t_count,'max_channel_sde':max_sde,
         'certifying_entry':witness,'sde_distribution':dict(sorted(stats.items())),
         'T_count_optimal':max_sde==t_count}




def _is_linearly_independent_paulis(labels):
    basis = {}
    for label in labels:
        chars = str(label)
        bits = tuple(int(c in 'XY') for c in chars) + tuple(int(c in 'YZ') for c in chars)
        v = sum(b << i for i, b in enumerate(bits))
        while v:
            leading = v.bit_length() - 1
            if leading in basis:
                v ^= basis[leading]
            else:
                basis[leading] = v
                break
        else:
            return False
    return True


def _label_symplectic(a, b):
    if len(a) != len(b):
        raise ValueError("Mismatched Pauli label lengths")
    xa = [c in 'XY' for c in a]
    za = [c in 'YZ' for c in a]
    xb = [c in 'XY' for c in b]
    zb = [c in 'YZ' for c in b]
    return sum(int(xa[i] and zb[i]) + int(za[i] and xb[i])
               for i in range(len(a))) % 2


def _four_axis_common_frame(a, b):
    """Check the independent-subset symplectic Gram criterion.

    The result is a Clifford existence claim, not a concrete Clifford circuit,
    and does not assert any single ZX rewrite changes only one event.
    """
    if len(a) != len(b) or len(a) < 4:
        return {"status":"not_applicable"}
    for ai in itertools.combinations(range(len(a)), 4):
        la = [a[i] for i in ai]
        if not _is_linearly_independent_paulis(la):
            continue
        ga = tuple(tuple(_label_symplectic(x,y) for y in la) for x in la)
        for bi in itertools.permutations(range(len(b)), 4):
            lb = [b[i] for i in bi]
            if not _is_linearly_independent_paulis(lb):
                continue
            gb = tuple(tuple(_label_symplectic(x,y) for y in lb) for x in lb)
            if ga == gb:
                return {"status":"CERTIFIED", "indices_A":list(ai),
                        "indices_B":list(bi), "Paulis_A":la, "Paulis_B":lb,
                        "joint_symplectic_gram":ga,
                        "interpretation":"4 projective Pauli axes admit a common Clifford mapping up to signs; this is not a gate-level ZX trace"}
    return {"status":"NOT_FOUND"}


def _read_witness_branch(result_dir, strategy):
    p = result_dir / "instances" / EXPECTED_WITNESS_CIRCUIT / strategy
    report_path = p / "variant_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"No successfully analyzed witness branch: {report_path}")
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    events = list(csv.DictReader((p / "pauli_events.csv").open(encoding="utf-8",newline="")))
    graph = nx.read_graphml(p / "anticommutation_graph.graphml")
    graph = nx.relabel_nodes(graph, {v: int(v) for v in graph.nodes()})
    ct = (p / "rebased_ct.qasm").read_text(encoding="utf-8")
    sigs = [r["pauli"] for r in events]
    for i,j in itertools.combinations(range(len(sigs)), 2):
        if graph.has_edge(i,j) != bool(_label_symplectic(sigs[i],sigs[j])):
            raise AssertionError(f"Pauli event and GraphML edge disagree: {strategy} {i},{j}")
    cert=rep.get("coloring_certificate",{})
    row=rep["row"]
    if not (row.get("equiv_status") == "pass" and
            row.get("analysis_success") is True and
            cert.get("proper_coloring_valid") is True and
            cert.get("hand_checkable_exact") is True and
            int(cert.get("clique_size",-1)) == int(row.get("chi",-2))):
        raise AssertionError(f"Uncertified witness branch: {strategy}")
    return {"directory":p, "report":rep, "events":events, "graph":graph,
            "qasm":ct, "pauli_labels":sigs, "chi":int(row["chi"]),
            "T_count":int(row["ct_t_count"]),
            "clique":cert.get("maximum_clique_witness",[])}


def _export_3graph_figure(branches, dst):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        fig, axes = plt.subplots(1,3,figsize=(14.5,4.5))
        for ax, (name, br) in zip(axes, branches.items()):
            g=br['graph']
            pos=nx.spring_layout(g,seed=42) if g.number_of_nodes() else {}
            color_obj=br['report'].get('coloring',{})
            colors=[int(color_obj.get(str(v),color_obj.get(v,0))) for v in g.nodes()]
            palette=['#2f8f85','#d49245','#8170ba','#5a8db5','#bf5b65','#9a9b61']
            nx.draw_networkx(g,pos=pos,ax=ax,with_labels=False,node_size=1000,
                             edge_color='#64748b',width=2.0,
                             node_color=[palette[c%len(palette)] for c in colors])
            for v,(x,y) in pos.items():
                ax.text(x,y,str(v)+':'+br['pauli_labels'][v],color='white',
                        ha='center',va='center',fontsize=8,fontweight='bold')
            ax.set_title(f"{name.replace('pyzx_','')}  |  T={br['T_count']}, "
                         f"omega={len(br['clique'])}, chi={br['chi']}")
            ax.axis('off')
        fig.tight_layout()
        for ext in ['png','pdf']:
            fig.savefig(dst.with_suffix('.'+ext),dpi=190,bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as exc:
        (dst.parent/'figure_warning.txt').write_text(repr(exc),encoding='utf-8')
        return False


def _zx_graph_snapshot(g, label, root):
    """Export ACTUAL regenerated ZX graph, not the generic ZX gate legend."""
    ensure_dir(root)
    coords, nodes, edges = {}, [], []
    def read_attr(attr, v, default=None):
        try:
            p = getattr(g, attr, None)
            if p is None: return default
            return p(v) if callable(p) else default
        except Exception:
            return default
    for v in g.vertices():
        vid=int(v)
        row=read_attr('row',v,vid)
        qb=read_attr('qubit',v,0)
        coords[vid]=(float(row),-float(qb))
        nodes.append({"id":vid, "row":str(row), "qubit":str(qb),
                      "type":str(read_attr('type',v)),
                      "phase":str(read_attr('phase',v))})
    for e in g.edges():
        u,v=(int(x) for x in tuple(e))
        edges.append({"u":u,"v":v,"edge_type":str(read_attr('edge_type',e))})
    obj={"label":label,"vertices":nodes,"edges":edges,
         "vertex_count":len(nodes),"edge_count":len(edges)}
    json_dump(root/(label+'.json'),obj)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(13,5))
        for edge in edges:
            u,v=edge['u'],edge['v']
            x0,y0=coords[u];x1,y1=coords[v]
            is_h=('HAD' in edge['edge_type'].upper() or
                  edge['edge_type'] in ('2','EdgeType.H'))
            ax.plot([x0,x1],[y0,y1],color=('#8560ad' if is_h else '#708190'),
                    linestyle=('--' if is_h else '-'),linewidth=0.85,zorder=1)
        for node in nodes:
            x,y=coords[node['id']]
            t=node['type'].upper()
            color='#239957' if ('Z' in t or t=='1') else ('#cd6767' if ('X' in t or t=='2') else 'white')
            ax.scatter([x],[y],s=48,facecolors=color,edgecolors='black',linewidths=.6,zorder=2)
            phase=node['phase']
            if phase not in ('None','0','0.0','0/1'):
                ax.text(x,y+0.18,phase,fontsize=6,ha='center')
        ax.set_title(f"{label} | {len(nodes)} vertices, {len(edges)} edges")
        ax.set_aspect('equal',adjustable='datalim')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(root/(label+'.svg'),bbox_inches='tight')
        fig.savefig(root/(label+'.png'),dpi=150,bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        obj['image_export_warning']=repr(exc)
        json_dump(root/(label+'.json'),obj)
    return obj


def _fresh_zx_snapshots(original, modes, branches, witness_dir):
    if not HAVE_PYZX:
        return {"status":"UNAVAILABLE", "reason":"PyZX not installed"}
    summary={}
    for st,mode in modes.items():
        outdir=witness_dir/st/'ZX_diagrams'
        ensure_dir(outdir)
        try:
            op=[]
            g=zx.Circuit.from_qasm(circuit_to_qasm2(original)).to_graph()
            a=_zx_graph_snapshot(g,'01_raw_input',outdir)
            g,_,_=pyzx_common_graphlike_preprocess(g,op)
            b=_zx_graph_snapshot(g,'02_common_graphlike',outdir)
            g=_apply_pyzx_strategy(g,mode,random.Random(7),1,op)
            c=_zx_graph_snapshot(g,'03_post_rewrite',outdir)
            normalized=_pyzx_extract_to_qiskit(g,op)
            normalized=qiskit_basis_preserving_transpile(normalized,0)
            ct=exact_ct_rebase(normalized)
            labels=[e.pauli.label() for e in extract_pauli_events(ct)]
            ref_labels=branches[st]['pauli_labels']
            equiv=operator_equiv_up_to_phase(original,normalized,max_qubits=8)
            branch_matches=(labels==ref_labels)
            status=('PASS' if equiv is True and branch_matches else 'MISMATCH')
            summary[st]={"status":status,"equivalent_to_original":equiv,
                         "Pauli_event_list_matches_main_run":branch_matches,
                         "post_rewrite_size":[c['vertex_count'],c['edge_count']],
                         "common_size":[b['vertex_count'],b['edge_count']],
                         "original_size":[a['vertex_count'],a['edge_count']],
                         "operations":op}
        except Exception as exc:
            summary[st]={"status":"ERROR","error":repr(exc),
                         "traceback":traceback.format_exc()[-3500:]}
    json_dump(witness_dir/'fresh_ZX_snapshot_validation.json',summary)
    return summary


def _draw_complete_circuit(qasm, output_path):
    try:
        if not HAVE_QISKIT:
            raise ImportError('Qiskit absent; using independent QASM figure')
        qc=circuit_from_qasm2(qasm)
        (output_path.with_suffix('.txt')).write_text(
             str(qc.draw(output='text',fold=105)),encoding='utf-8')
        fig=qc.draw(output='mpl',fold=25,idle_wires=True)
        fig.savefig(output_path,dpi=170,bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)
        return 'PASS'
    except Exception as exc:
        # Qiskit Matplotlib drawings may require optional pylatexenc. Reconstruct
        # an exact, complete sequential gate diagram directly from QASM instead.
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            n,ops=parse_qasm(qasm)
            per_row=22
            chunks=[ops[i:i+per_row] for i in range(0,len(ops),per_row)]
            fig,ax=plt.subplots(figsize=(15,max(2.8,2.0*len(chunks))))
            ti=0
            for rowno,chunk in enumerate(chunks):
                offset=rowno*(n+1.3)
                for q in range(n):
                    y=offset+q
                    ax.plot([0.5,per_row+1],[y,y],color='#94a3b8',linewidth=1)
                    ax.text(0.2,y,f'q{q}',va='center',ha='right',fontsize=9)
                for j,(gate,qs) in enumerate(chunk,1):
                    x=j+0.05
                    if gate=='cx':
                        c,t=qs
                        yy1,yy2=offset+c,offset+t
                        ax.plot([x,x],[yy1,yy2],color='#374151',lw=1.5)
                        ax.plot(x,yy1,'o',ms=4,color='#374151')
                        ax.plot(x,yy2,'o',ms=13,mfc='white',mec='#374151')
                        ax.plot([x-.2,x+.2],[yy2,yy2],color='#374151',lw=1)
                        ax.plot([x,x],[yy2-.2,yy2+.2],color='#374151',lw=1)
                    else:
                        q=qs[0]
                        tlike=gate in ('t','tdg')
                        label=(f'{gate.upper()}\nT{ti}' if tlike else gate.upper())
                        if tlike:ti+=1
                        ax.text(x,offset+q,label,ha='center',va='center',fontsize=7,
                           bbox=dict(facecolor=('#3b82f6' if tlike else '#f1f5f9'),
                                     edgecolor='#334155',boxstyle='round,pad=.22'),
                           color=('white' if tlike else 'black'))
            ax.set_xlim(0,per_row+1.3)
            ax.set_ylim(-.5,len(chunks)*(n+1.3))
            ax.invert_yaxis()
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(output_path,dpi=170,bbox_inches='tight')
            fig.savefig(output_path.with_suffix('.pdf'),bbox_inches='tight')
            plt.close(fig)
            return 'PASS_FALLBACK_COMPLETE_GATE_SEQUENCE'
        except Exception as fallback_exc:
            return 'IMAGE_UNAVAILABLE: '+repr(exc)+'; fallback='+repr(fallback_exc)


def _write_k_coloring_dimacs(graph, k, path):
    nodes=sorted(int(v) for v in graph.nodes())
    idx={node:j for j,node in enumerate(nodes)}
    def v(node,c):return 1 + idx[int(node)]*k + c
    clauses=[]
    for node in nodes:
        clauses.append([v(node,c) for c in range(k)])
        for a in range(k):
            for b in range(a+1,k):
                clauses.append([-v(node,a),-v(node,b)])
    for a,b in graph.edges():
        for c in range(k):
            clauses.append([-v(a,c),-v(b,c)])
    lines=[f"p cnf {len(nodes)*k} {len(clauses)}"]
    lines.extend(' '.join(str(x) for x in clause)+' 0' for clause in clauses)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {"variables":len(nodes)*k,"clauses":len(clauses)}


def produce_canonical_witness_evidence(result_dir):
    """Derive ALL witness evidence from freshly produced QASM files.

    No external archive is read; witness and channel claims FAIL CLOSED.
    """
    witness_dir=result_dir/'paper_witness'
    ensure_dir(witness_dir)
    summary={"created_at":datetime.now(timezone.utc).isoformat(),
             "circuit_id":EXPECTED_WITNESS_CIRCUIT,"profile":RUN_PROFILE,
             "expected_chi":EXPECTED_WITNESS_CHI,"branches":{}}
    try:
        branches={name:_read_witness_branch(result_dir,name)
                  for name in EXPECTED_WITNESS_CHI}
        channels={}
        for name, branch in branches.items():
            outdir=witness_dir/name
            ensure_dir(outdir)
            qasm=branch['qasm']
            n,ops=parse_qasm(qasm)
            if n != 4 or sum(g in ('t','tdg') for g,_ in ops) != 5:
                raise AssertionError('Canonical witness must have 4 qubits and 5 explicit T events')
            cert=verify(qasm)
            channel=exact_channel(qasm)
            channels[name]=channel
            if cert['max_channel_sde'] != 5 or not cert['T_count_optimal']:
                raise AssertionError('Five-T minimality not certified')
            if branch['T_count']!=5 or branch['chi']!=EXPECTED_WITNESS_CHI[name]:
                raise AssertionError('Unexpected witness T-count or chromatic number: '+name)
            G=branch['graph']
            for k in range(1,branch['chi']+1):
                _write_k_coloring_dimacs(G,k,outdir/f"{k}_coloring.cnf")
            drawing=_draw_complete_circuit(qasm,outdir/'complete_CT_circuit.png')
            (outdir/'rebased_ct.qasm').write_text(qasm,encoding='utf-8')
            summary['branches'][name]={"status":"PASS","T_count":branch['T_count'],
                "chi":branch['chi'],"edges":int(G.number_of_edges()),
                "omega":len(branch['clique']),"T_count_certificate":cert,
                "channel_nonzero_entries":len(channel),
                "complete_CT_circuit_image":drawing}
        base=channels['pyzx_no_simp']
        exact_same=all(channel==base for channel in channels.values())
        if not exact_same:
            raise AssertionError('Exact Pauli-transfer matrices are not identical')
        summary['exact_projective_unitary_equality']=True
        summary['certified_min_ancilla_free_T_count']=5
        summary['channel_matrix_dimension']=4**4
        summary['distinct_nonzero_channel_entries']=len(base)
        summary['all_three_chi']=[branches[s]['chi'] for s in EXPECTED_WITNESS_CHI]
        summary['clifford_frame_four_axis_comparison']=_four_axis_common_frame(
            branches['pyzx_no_simp']['pauli_labels'],
            branches['pyzx_clifford_simp']['pauli_labels'])
        summary['three_graphs_drawn']=_export_3graph_figure(
            branches,witness_dir/'three_Pauli_graphs')
        # Reproduce actual ZX intermediate graphs using a SEPARATE fresh path.
        if SAVE_ZX_WITNESS_SNAPSHOTS:
            orig_path=result_dir/'instances'/EXPECTED_WITNESS_CIRCUIT/'original.qasm'
            original=circuit_from_qasm2(orig_path.read_text(encoding='utf-8'))
            summary['fresh_ZX_snapshots']=_fresh_zx_snapshots(original,{
                'pyzx_no_simp':'none','pyzx_clifford_simp':'clifford_simp',
                'pyzx_pivot_only':'pivot_only'},branches,witness_dir)
            if not all(v.get('status')=='PASS' for v in summary['fresh_ZX_snapshots'].values()):
                summary['ZX_snapshots_match']=False
                summary['warnings']=['At least one fresh ZX trace differs from saved branch']
            else:
                summary['ZX_snapshots_match']=True
        summary['status']='PASS' if summary.get('ZX_snapshots_match',True) else 'PARTIAL'
    except Exception as exc:
        summary['status']='FAIL'
        summary['error']=repr(exc)
        summary['traceback']=traceback.format_exc()[-5500:]
    json_dump(witness_dir/'EXACT_WITNESS_CERTIFICATE.json',summary)
    return summary


def _paper_family_table(result_dir):
    p=result_dir/'variant_metrics.csv'
    if not p.exists():return None
    df=pd.read_csv(p)
    base=pd.read_csv(result_dir/'benchmark_manifest.csv')
    fam=pd.read_csv(result_dir/'family_sensitivity_summary.csv') if (result_dir/'family_sensitivity_summary.csv').exists() else pd.DataFrame()
    valid=df[(df['analysis_success']==True)&(df['equiv_status']=='pass')].copy()
    rows=[]
    for family, grp in base.groupby('family'):
        d=valid[valid['family']==family]
        fs=fam[fam['family']==family] if not fam.empty else pd.DataFrame()
        rows.append({
          'family':family, 'circuits':len(grp),
          'min_qubits':int(grp['qubits'].min()),'max_qubits':int(grp['qubits'].max()),
          'min_input_ops':int(grp['ops'].min()),'max_input_ops':int(grp['ops'].max()),
          'min_original_T_count':int(pd.to_numeric(grp['original_CT_T_count'],errors='coerce').min()) if grp['original_CT_T_count'].notna().any() else None,
          'max_original_T_count':int(pd.to_numeric(grp['original_CT_T_count'],errors='coerce').max()) if grp['original_CT_T_count'].notna().any() else None,
          'T_matched_chromatic_sensitive':int(fs['T_matched_chromatic_sensitive'].iloc[0]) if not fs.empty else None,
          'valid_branches':len(d),
          'median_exact_coloring_seconds':float(pd.to_numeric(d['exact_coloring_runtime_s'],errors='coerce').median()) if not d.empty else None,
          'max_graph_vertices':int(d['graph_nodes'].max()) if not d.empty else None,
          'max_graph_edges':int(d['graph_edges'].max()) if not d.empty else None,
          'median_SAT_variables':float(pd.to_numeric(d['sat_chi_variables'],errors='coerce').median()) if not d.empty else None,
          'median_SAT_clauses':float(pd.to_numeric(d['sat_chi_clauses'],errors='coerce').median()) if not d.empty else None,
        })
    result=pd.DataFrame(rows).sort_values('family')
    out=result_dir/'paper_ready'
    ensure_dir(out)
    result.to_csv(out/'benchmark_family_size_runtime_table.csv',index=False)
    result[['family','circuits','min_qubits','max_qubits',
            'min_original_T_count','max_original_T_count',
            'T_matched_chromatic_sensitive','median_exact_coloring_seconds']].to_latex(
        out/'benchmark_family_size_runtime_table.tex',index=False,
        float_format='%.4f',escape=True)
    return result


def _paper_plots(result_dir,family_table):
    if family_table is None or family_table.empty or not EXPORT_PAPER_FIGURES:
        return []
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        out=result_dir/'paper_ready'
        names=[]
        data=family_table.sort_values('T_matched_chromatic_sensitive',ascending=False)
        fig,ax=plt.subplots(figsize=(13,5))
        ax.bar(data['family'],data['T_matched_chromatic_sensitive'])
        ax.set_ylabel('T-count-matched chromatically sensitive circuits')
        ax.set_xlabel('Benchmark family (10 circuits each in paper profile)')
        ax.set_title('Exact Pauli-graph chromatic sensitivity by family')
        ax.tick_params(axis='x',labelrotation=55)
        fig.tight_layout()
        for ext in ['png','pdf']:
            name='family_matched_chromatic_sensitivity.'+ext
            fig.savefig(out/name,dpi=190,bbox_inches='tight');names.append(name)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5))
        df=pd.read_csv(result_dir/'variant_metrics.csv')
        valid=df[(df['analysis_success']==True)&(df['equiv_status']=='pass')]
        ax.scatter(valid['ct_t_count'],valid['chi'],s=9,alpha=.3)
        ax.set_xlabel('Explicit post-rebase T count')
        ax.set_ylabel('Chromatic number of Pauli anticommutation graph')
        ax.set_title('Chi is an algebraic commuting-class descriptor, not T depth')
        fig.tight_layout()
        for ext in ['png','pdf']:
            name='T_count_vs_chi.'+ext
            fig.savefig(out/name,dpi=190,bbox_inches='tight');names.append(name)
        plt.close(fig)
        return names
    except Exception as exc:
        (result_dir/'paper_ready'/'plot_warning.txt').write_text(repr(exc),encoding='utf-8')
        return []


def _paper_claim_consistency(result_dir, witness):
    core=json.loads((result_dir/'publication_scale_conclusion.json').read_text(encoding='utf-8'))
    keys={'benchmarks':220,'families':22,'strategies':13,
          'planned_strategy_level_branches':2860,'valid_equivalent_variants':2842,
          'T_count_matched_graph_sensitive_circuits':61,
          'T_count_matched_chromatic_sensitive_circuits':39}
    rows={key:{'expected':expected,'observed':core.get(key),
               'match':core.get(key)==expected} for key,expected in keys.items()}
    # These are comparisons to the fixed manuscript data, NOT forced results.
    # A witness/quick run does not contain 220 circuits; comparing its
    # totals with the paper benchmark is not a scientific mismatch.
    # Report the scope explicitly while keeping the witness result separate.
    if RUN_PROFILE == 'paper':
        paper_status = ('MATCH' if all(x['match'] for x in rows.values()) and
                        witness.get('status') == 'PASS' else 'MISMATCH')
    else:
        paper_status = 'NOT_APPLICABLE_PARTIAL_PROFILE'
    results={"profile":RUN_PROFILE,"benchmarks":rows,
      "status":paper_status,
      "witness_status":witness.get('status'),
      "warning":"MISMATCH does not mean the new run is invalid. Determine dependency versions, normalization/CT exclusions, and benchmark selection before comparing with the published 2026 dataset.",
      "expected_historical_run_failure_counts":{
          "equivalence_failed":2,"exact_CT_model_out_of_domain":16},
      "not_claimed":"These counts are dependent on the specified tool versions and accepted-branch policy."}
    json_dump(result_dir/'PAPER_CLAIM_VALIDATION.json',results)
    return results




def independent_pauli_from_qasm(qasm):
    """Second implementation of suffix Clifford propagation, with integer bitsets.

    Unlike extract_pauli_events(), this method constructs no numpy matrices.
    In the reverse gate scan, images of the 2n Pauli basis vectors under the
    current Clifford suffix are updated using only XOR and column swaps.
    It is projective (Pauli signs are irrelevant for anticommutation).
    """
    n, ops = parse_qasm(qasm)
    columns = [1 << j for j in range(2*n)]
    events = []
    for gate, qs in reversed(ops):
        if gate in ('t','tdg'):
            q=qs[0]
            bits=columns[n+q]
            letters=[]
            for k in range(n):
                x=(bits>>k)&1
                z=(bits>>(n+k))&1
                letters.append('I' if not x and not z else
                               'X' if x and not z else
                               'Z' if z and not x else 'Y')
            events.append((gate,q,''.join(letters)))
        elif gate=='h':
            q=qs[0]
            columns[q],columns[n+q]=columns[n+q],columns[q]
        elif gate in ('s','sdg'):
            q=qs[0]
            columns[q] ^= columns[n+q]
        elif gate=='cx':
            c,t=qs
            columns[c] ^= columns[t]
            columns[n+t] ^= columns[n+c]
        elif gate in ('x','z','id'):
            pass
        else:
            raise ValueError('Unsupported exact Clifford+T instruction in independent audit: '+gate)
    return n, list(reversed(events))


def audit_all_accepted_variants_from_qasm(result_dir):
    """Independent algorithmic check of EVERY accepted branch (no solver re-run).

    Explicitly distinguishes verified graph/color/clique witnesses from exact
    SAT optimum lower bounds: a SAT UNSAT verdict is not a proof log.
    """
    d=pd.read_csv(result_dir/'variant_metrics.csv')
    valid=d[(d['analysis_success']==True)&(d['equiv_status']=='pass')].copy()
    findings=[]
    print(f'  Independent audit of {len(valid)} in-domain, equivalent branches...',flush=True)
    for count,(_,r) in enumerate(valid.iterrows(),1):
        cid,st=str(r['circuit_id']),str(r['strategy'])
        branch=result_dir/'instances'/cid/st
        rec={'circuit_id':cid,'strategy':st,'family':str(r['family']),
             'status':'FAIL','labels_match_QASM':False,'edges_match_QASM':False,
             'proper_coloring':False,'clique_verified':False,
             'clique_proves_chi':False}
        try:
            qasm=(branch/'rebased_ct.qasm').read_text(encoding='utf-8')
            n,events=independent_pauli_from_qasm(qasm)
            event_csv=list(csv.DictReader((branch/'pauli_events.csv').open(encoding='utf-8',newline='')))
            labels=[e[2] for e in events]
            rec['qubits']=n
            rec['T_count']=len(labels)
            rec['labels_match_QASM']=(len(event_csv)==len(events) and all(
                str(e['pauli'])==sig and str(e['gate'])==gate and int(e['qubit'])==q
                for e,(gate,q,sig) in zip(event_csv,events)))
            g=nx.read_graphml(branch/'anticommutation_graph.graphml')
            g=nx.relabel_nodes(g,{v:int(v) for v in g.nodes()})
            expected={ (i,j) for i in range(len(labels)) for j in range(i+1,len(labels))
                       if _label_symplectic(labels[i],labels[j])}
            observed={tuple(sorted((int(a),int(b)))) for a,b in g.edges()}
            rec['edges_match_QASM']=(observed==expected and
                                    set(g.nodes())==set(range(len(labels))))
            cert=json.loads((branch/'variant_report.json').read_text(encoding='utf-8'))
            colors={int(i):int(c) for i,c in cert['coloring'].items()}
            chi=int(r['chi'])
            rec['proper_coloring']=(len(colors)==len(labels) and
                set(colors)==set(range(len(labels))) and
                (1+max(colors.values(),default=-1)==chi) and
                all(colors[i]!=colors[j] for i,j in expected))
            witness=[int(v) for v in cert['coloring_certificate'].get('maximum_clique_witness',[])]
            rec['clique_verified']=(len(set(witness))==len(witness) and
                all(tuple(sorted((i,j))) in expected for i,j in itertools.combinations(witness,2)))
            rec['clique_size']=len(witness)
            rec['reported_chi']=chi
            rec['clique_proves_chi']=(len(witness)==chi and rec['clique_verified']
                                       and rec['proper_coloring'])
            rec['status']='PASS' if all(rec[x] for x in (
                'labels_match_QASM','edges_match_QASM','proper_coloring','clique_verified')) else 'FAIL'
        except Exception as exc:
            rec['error']=repr(exc)
        findings.append(rec)
        if count%250==0 or count==len(valid):
            print(f'   audited {count}/{len(valid)} | failures='
                  f'{sum(z["status"]!="PASS" for z in findings)}',flush=True)
    res=pd.DataFrame(findings)
    out=result_dir/'paper_ready'
    ensure_dir(out)
    res.to_csv(out/'all_accepted_branches_independent_Pauli_audit.csv',index=False)
    counts={'analyzed':len(res),
            'all_independent_graph_checks_pass':bool((res['status']=='PASS').all()) if not res.empty else False,
            'failures':int((res['status']!='PASS').sum()) if not res.empty else 0,
            'clique_and_coloring_optima_certified':int(res['clique_proves_chi'].sum()) if not res.empty else 0,
            'note':'For non-clique-tight graphs, the original exact search and SAT UNSAT verdict support optimality; this independent audit does not manufacture UNSAT proof logs.'}
    json_dump(out/'INDEPENDENT_AUDIT_SUMMARY.json',counts)
    return counts


def finish_single_file_reproduction(result_dir):
    print('\n[IJQI] Producing complete publication evidence...',flush=True)
    result_dir=Path(result_dir)
    witness=produce_canonical_witness_evidence(result_dir)
    print('  Canonical three-strategy witness:',witness['status'],flush=True)
    family=_paper_family_table(result_dir)
    plots=_paper_plots(result_dir,family)
    audit=audit_all_accepted_variants_from_qasm(result_dir)
    print('  Independent Pauli audit:',audit['analyzed'],'branches,',
          audit['failures'],'mismatches,',
          audit['clique_and_coloring_optima_certified'],'clique-tight.',flush=True)
    if RUN_PROFILE=='paper':
        consistency=_paper_claim_consistency(result_dir,witness)
        consistency['independent_audit']=audit
        if not audit['all_independent_graph_checks_pass']:
            consistency['status']='MISMATCH'
        json_dump(result_dir/'PAPER_CLAIM_VALIDATION.json',consistency)
        print('  Original manuscript numeric claims:',consistency['status'],flush=True)
    else:
        consistency={"status":"NOT_APPLICABLE_TO_SUBSET","profile":RUN_PROFILE}
        json_dump(result_dir/'PAPER_CLAIM_VALIDATION.json',consistency)
    README=f"""IJQI / ZX normalization sensitivity -- publication output

Mode: {RUN_PROFILE}
Scripts required: ONE (zx_normalization_sensitivity.py)

The paper profile requires Qiskit, PyZX, python-sat and mqt.qcec; matplotlib
and pylatexenc are recommended for publication figures. The program deliberately
fails early if proof or equivalence backends are missing in paper mode.

Core: V8.4 Pauli anticommutation model; chi is NOT physical T depth.

variant_metrics.csv: all attempts incl. rejected/unknown with reasons
circuit_summary.csv, family_sensitivity_summary.csv: family and circuit analyses
T_count_matched_pairs.csv: strict same-T pairs
benchmark_manifest.csv, benchmark_family_design.csv: benchmark details
strategy_saturation.csv, leave_one_strategy_out.csv: robustness
performance_summary.csv, failure_forensics.csv, strategy_health.csv: run costs/failures
instances/: full QASM, graphml, Pauli events, coloring certificates per branch
paper_witness/: fresh three-branch witness QASM, ZX snapshots, exact-channel proof,
                exact denominator-T lower bound, clique/coloring and SAT CNFs
paper_ready/: family runtime tables, plots, all-branch independent Pauli audit
PAPER_CLAIM_VALIDATION.json: original paper-number comparisons (never altered)

Witness status: {witness.get('status')}
Claim-consistency status: {consistency.get('status')}
Paper plots: {plots}

Do not present a partial run, absent QCEC/unknown status or a numerical
T-count lower bound as an independently verified all-branch proof.
"""
    (result_dir/'OUTPUT_README.txt').write_text(README,encoding='utf-8')
    # Create a compact review bundle; leave raw per-branch files in the folder.
    zpath=result_dir/'IJQI_KEY_PUBLICATION_EVIDENCE.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(result_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in ('.json','.csv','.txt'):
                z.write(p,arcname=p.name)
        for dirname in ['paper_witness','paper_ready','PRIMARY_WITNESS']:
            folder=result_dir/dirname
            if folder.exists():
                for p in folder.rglob('*'):
                    if p.is_file():
                        z.write(p,arcname=str(p.relative_to(result_dir)))
    return {"witness_status":witness['status'],
            "claim_validation":consistency['status'],
            "outputs":str(result_dir.resolve()),"paper_figures":plots,
            "independent_audit":audit}


if __name__ == '__main__':
    print('='*72)
    print('ZX NORMALIZATION SENSITIVITY | IJQI SINGLE-FILE REPRODUCTION')
    print('Profile:',RUN_PROFILE,'| Script:',Path(__file__).resolve())
    print('Output:',OUTPUT_DIR)
    print('='*72,flush=True)
    main()  # The full validated V8.4 pipeline, without external helper modules.
    report=finish_single_file_reproduction(OUTPUT_DIR)
    print('\nPUBLICATION OUTPUT STATUS:',json.dumps(report,indent=2),flush=True)
