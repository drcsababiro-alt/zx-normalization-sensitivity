# Chromatic Sensitivity at Optimal T-Count in ZX-Normalized Clifford+T Representations

**Reproducibility repository**  
**Author:** Csaba Biró  
**Research area:** quantum circuit optimization, ZX-calculus, Clifford+T representations, Pauli anticommutation graphs, and exact graph coloring

## Overview

This repository provides the single-file Python implementation and complete experiment data for the study **“Chromatic Sensitivity at Optimal T-Count in ZX-Normalized Clifford+T Representations.”**

The central question is whether an algebraic graph descriptor of non-Clifford events is preserved across semantically equivalent circuit representations produced by different ZX-calculus normalization and extraction paths. Each resulting circuit is rebased into an exact Clifford+T gate set, and every explicit `T` or `Tdg` event is assigned a Pauli signature in the Clifford frame. These signatures define a **Pauli-anticommutation graph**, whose exact chromatic number is compared across representations.

The study shows that **even representations attaining the same minimum ancilla-free T-count can have different graph chromatic numbers**.

## Main results

| Quantity | Result |
| --- | ---: |
| Benchmark circuits | 220 |
| Circuit families | 22 |
| Normalization/compiler strategies | 13 |
| Strategy-level attempts | 2,860 |
| Equivalence-verified, exact-model-eligible representations | 2,842 |
| Equivalent branches outside the supported exact Clifford+T rebasing domain | 18 |
| Circuits with graph variation at matched T-count | 61 |
| Circuits with chromatic variation at matched T-count | **39** |
| Independently audited accepted Pauli graphs | 2,842 |
| Graphs with matching clique and coloring certificates | 2,685 |

The four-qubit witness `VQE_slot07_vqe_su2_n4_r3_s7` is represented by three PyZX paths:

| ZX processing path | Explicit T/Tdg events | Chromatic number |
| --- | ---: | ---: |
| `pyzx_no_simp` | 5 | 3 |
| `pyzx_clifford_simp` | 5 | 2 |
| `pyzx_pivot_only` | 5 | 3 |

The three witness circuits implement the same unitary **up to global phase**. Exact Pauli-transfer calculations yield a T-count lower bound of five in the **ancilla-free, measurement-free Clifford+T model**, making all three five-T realizations optimal within that model. The graph chromatic numbers are certified by explicit cliques and proper colorings.

## Mathematical model

For a fixed exactly rebased circuit, let the explicit non-Clifford events be indexed by `i = 0, ..., m-1`. Propagation through the Clifford suffix assigns each event a projective Pauli signature `P_i`, represented by binary symplectic vectors `(x_i, z_i)`.

Two distinct event vertices are connected precisely when their Pauli signatures anticommute:

$$
\{i,j\}\in E(G_{\mathrm{AC}})
\quad\Longleftrightarrow\quad
x_i\cdot z_j+z_i\cdot x_j=1\pmod{2}.
$$

A proper coloring of this graph partitions the fixed event list into classes of pairwise-commuting Pauli signatures. Thus `chi(G_AC)` is the **minimum number of such commuting classes** for the chosen representation.

**Scope:** `chi(G_AC)` is *not* asserted to be physical T-depth, executable scheduling depth, or an invariant of the unitary. The comparison is relative to the specified circuit-extraction and rebasing pipeline. The tested normalization strategies constitute a finite sample, not all possible ZX derivations.

## Repository files

```text
.
├── README.md
├── zx_normalization_sensitivity.py
└── datas.zip
```

- **`zx_normalization_sensitivity.py`** — self-contained experiment driver, including benchmark generation, ZX processing, equivalence verification, exact Clifford+T rebasing, graph extraction, coloring, robustness analysis, the exact witness calculation, and publication-data export. No additional project-local Python modules are required.
- **`datas.zip`** — complete archived output of the `paper` profile, including source circuits, extracted and rebased QASM, Pauli-event and edge tables, GraphML graphs, exact-coloring information, independent audit results, witness certificates, and publication figures. The ZIP extracts to **`zx_normalization_sensitivity_output_paper/`**.

The archived results can be examined **without rerunning** the full experiment.

## Installation

Python **3.10** was used for the archived experiment. Use a dedicated Python environment (the packages must be installed in the interpreter selected by Spyder if running there).

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas networkx matplotlib qiskit pyzx python-sat mqt.qcec pylatexenc
```

For the complete `paper` profile, **Qiskit, PyZX, python-sat, and MQT QCEC are required**; the program checks for missing verification backends before executing the study. Matplotlib and `pylatexenc` support publication graphics. `pytket-qiskit` is optional, and its strategy is disabled by the default configuration (`INCLUDE_TKET = False`).

Selected recorded environment details are available in `environment.json` inside `datas.zip` (Python 3.10.20; NumPy 2.2.6; pandas 2.3.3; NetworkX 3.4.2). The ZX witness snapshots were generated using PyZX 0.10.2. Other package versions may affect compiler outputs and consequently the benchmark counts; inspect the archived environment and run metadata when comparing independent executions.

## Running the experiment

Open `zx_normalization_sensitivity.py` in Spyder and press **F5**, or run it from a terminal:

```bash
python zx_normalization_sensitivity.py
```

The script is configured in its **CONFIG** section, without command-line arguments:

```python
RUN_PROFILE = "paper"  # "paper", "witness", or "quick"
```

| Profile | Purpose | Output directory |
| --- | --- | --- |
| `witness` | Three PyZX paths for the exact four-qubit witness | `zx_normalization_sensitivity_output_witness/` |
| `quick` | Small functionality check | `zx_normalization_sensitivity_output_quick/` |
| `paper` | Full 220-circuit, 13-strategy study | `zx_normalization_sensitivity_output_paper/` |

**Recommended workflow:** first run `witness`, then run `paper`. Each run writes to a directory **next to the Python file**, independently of Spyder's current working directory. The full run may require substantial computation and memory.

**Important for independent reruns:** the archive is a completed result set. Do **not** extract it into the script directory immediately before attempting a fresh computation: with `RESUME_INTERRUPTED_RUN = True`, the script can reuse existing branch records. To reproduce the results independently, run the Python file in a **clean directory**, without an existing `zx_normalization_sensitivity_output_paper/` folder. Resume mode is intended for continuing an interrupted run with the **identical source file**.

## Inspecting the archived data

Extract `datas.zip` at repository root:

```bash
python -m zipfile -e datas.zip .
```

The resulting `zx_normalization_sensitivity_output_paper/` directory includes:

| Path | Contents |
| --- | --- |
| `PAPER_CLAIM_VALIDATION.json` | Main aggregate counts and consistency checks |
| `publication_scale_conclusion.json` | Experiment-level statistics and interpretation limits |
| `environment.json`, `run_integrity.json` | Execution metadata and source fingerprint |
| `benchmark_manifest.csv`, `benchmark_family_design.csv` | Circuit definitions and family composition |
| `variant_metrics.csv` | Per-strategy results, equivalence status, and exclusions |
| `circuit_summary.csv`, `family_sensitivity_summary.csv` | Circuit- and family-level summaries |
| `T_count_matched_pairs.csv` | Pairwise comparisons constrained to equal T-count |
| `strategy_health.csv`, `failure_forensics.csv` | Strategy coverage and excluded-branch diagnostics |
| `leave_one_strategy_out.csv`, `strategy_saturation.csv` | Finite-strategy robustness analyses |
| `instances/` | Per-circuit/per-strategy QASM, Pauli events, graph edges, GraphML, and metadata |
| `paper_witness/` | Three-path exact witness, ZX snapshots, circuits, graphs, and certificates |
| `paper_witness/EXACT_WITNESS_CERTIFICATE.json` | Exact witness equality, chromatic numbers, and ancilla-free T-count optimality |
| `paper_ready/INDEPENDENT_AUDIT_SUMMARY.json` | Cross-checks of accepted Pauli graph reconstructions |
| `paper_ready/` | Publication figures, family table, and branch-level independent audit |

Useful entry points after extraction are `PAPER_CLAIM_VALIDATION.json`, `publication_scale_conclusion.json`, the witness certificate, and `paper_ready/`. The repository intentionally keeps the full output in one ZIP archive, rather than publishing thousands of generated files individually.

### Validation notes

- All **2,842** accepted event graphs passed the independent reconstruction audit.
- **2,685** graphs have matching clique lower bounds and proper-coloring upper bounds. Exact solver verdicts support optimality for the remaining cases; separate externally checkable UNSAT proof logs are **not** supplied for every such case.
- The **18** excluded branches passed semantic equivalence checking but fell outside the supported exact Clifford+T rebasing domain. They were not used in the graph sensitivity analysis.
- The exact five-T optimality statement concerns the **ancilla-free Clifford+T** model; no claim is made about measurement-assisted or ancilla-assisted synthesis.

## Reproducibility and citation

The archived `paper` run records the SHA-256 fingerprint of its Python source:

```text
5ec6297fe08e08b50bce2f06e612131190f9ef478d1e94c0d4b2926f31df1d7e
```

To verify the source file locally:

```bash
python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('zx_normalization_sensitivity.py').read_bytes()).hexdigest())"
```

**Citation:**

> Csaba Biró. *Chromatic Sensitivity at Optimal T-Count in ZX-Normalized Clifford+T Representations*. 2026. Manuscript submitted to the *International Journal of Quantum Information*.

No DOI or publication details are asserted until assigned by the publisher.
