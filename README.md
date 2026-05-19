# ZX Normalization Sensitivity

Experimental framework for analyzing representation-sensitive dependency abstractions in ZX-normalized Clifford+T quantum circuits.

This repository accompanies the manuscript:

**Semantic Equivalence and Dependency-Graph Sensitivity in ZX-Normalized Clifford+T Circuits**

The code studies how different semantics-preserving normalization and extraction strategies may induce different T-dependency graphs and different SAT-certified graph-optimal T-layer values.

## Repository contents

```text
zx_normalization_sensitivity.py   Main experimental script
requirements.txt                  Python dependencies
README.md                         Reproducibility notes
CITATION.cff                      Citation metadata
.gitignore                        Suggested ignored outputs
```

Generated experiment folders are not required for running the script, but may be added separately under `results/` or as release artifacts.

## Main workflow

For an input Clifford+T circuit \(C\), the pipeline evaluates

```text
C -> N(C) -> G_T(N(C)) -> chi(G_T(N(C)))
```

where:

- `N(C)` is a normalized representation produced by a ZX/Qiskit/tket strategy,
- `G_T(N(C))` is the induced T-dependency graph,
- `chi(G_T(N(C)))` is the SAT-certified graph-optimal T-layer value.

The analysis records graph diversity, chromatic diversity, replay gaps, and family-level sensitivity summaries. Failed extraction or failed equivalence checks are quarantined and are not used as evidence for sensitivity.

## Installation

Python 3.10 is recommended.

```bash
pip install -r requirements.txt
```

Optional packages such as `pyzx`, `mqt.qcec`, `pytket`, `pynauty`, and `scipy` improve coverage or add additional backends, but the script is designed to fail softly when optional tools are unavailable.

## Example runs

Quick sanity check:

```bash
python zx_normalization_sensitivity.py --outdir artifacts_normsens --quick
```

Medium experiment:

```bash
python zx_normalization_sensitivity.py --outdir artifacts_normsens --medium --pairwise-equivalence
```

Larger experiment:

```bash
python zx_normalization_sensitivity.py --outdir artifacts_normsens --large --pairwise-equivalence --random-seeds 0 1 2 3 4
```

External OpenQASM 2 directory:

```bash
python zx_normalization_sensitivity.py --outdir artifacts_normsens --medium --qasm-dir path/to/qasm
```

## Main output files

The output directory contains several CSV, JSON, and graph artifacts. The most important summary files are:

```text
variant_metrics.csv
circuit_summary.csv
family_summary.csv
counterexamples.csv
strategy_health.csv
conclusion.json
summary_artifacts.zip
```

The sensitivity claims in the manuscript are based on variants with successful analysis and passing equivalence validation.

## Reproducibility note

The explored normalization families are finite computationally generated subsets of semantics-preserving rewriting pathways. Reported graph diversity and chromatic diversity values should therefore be interpreted as empirical lower bounds for the explored strategies and benchmark suite, not as exhaustive invariants over all possible ZX rewrites.

## License

A license file should be added before public release. MIT or Apache-2.0 are both suitable choices for this type of research code.


## Benchmark families and generation switches

The framework supports several generated and imported circuit families. Depending on the selected run profile, the script combines deterministic benchmark circuits with randomized structured instances.

Typical families include:

- Grover-like oracle circuits
- QAOA ring and path circuits
- QFT-derived structures
- Dense/brick interaction layouts
- Random Clifford+T circuits
- Random CNOT+T circuits
- Hardware-efficient ansätze
- EfficientSU2 and RealAmplitudes variants
- Modular-adder-style proxy circuits

Some families can be enabled or disabled explicitly from the command line.

Examples:

Enable Grover-like benchmarks:

```bash
python zx_normalization_sensitivity.py --medium --enable-grover
```

Disable Grover-like benchmarks:

```bash
python zx_normalization_sensitivity.py --medium --disable-grover
```

Restrict to QAOA-style circuits:

```bash
python zx_normalization_sensitivity.py --medium --only-qaoa
```

Enable randomized structured generators:

```bash
python zx_normalization_sensitivity.py --large --randomized-families
```

The exact switches available depend on the current script version and enabled optional libraries.

---

## Run profiles

The script provides several predefined workload profiles.

### Quick profile

Designed for sanity checks, dependency validation, and small local experiments.

```bash
python zx_normalization_sensitivity.py --quick
```

Characteristics:
- small circuit counts
- shallow benchmark depth
- reduced normalization branching
- shorter SAT solving limits
- suitable for laptops

---

### Medium profile

Balanced configuration intended for most reproducibility experiments.

```bash
python zx_normalization_sensitivity.py --medium --pairwise-equivalence
```

Characteristics:
- broader benchmark coverage
- moderate normalization branching
- stronger equivalence validation
- suitable for workstation-class hardware

---

### Large profile

Extended experimental configuration for larger benchmark suites.

```bash
python zx_normalization_sensitivity.py --large --pairwise-equivalence
```

Characteristics:
- larger randomized benchmark sets
- more normalization strategies
- increased SAT solving workload
- additional graph export artifacts
- intended for long-running experiments

---

## Equivalence validation options

The framework supports several equivalence validation modes.

Basic validation:

```bash
python zx_normalization_sensitivity.py --medium
```

Pairwise equivalence validation:

```bash
python zx_normalization_sensitivity.py --medium --pairwise-equivalence
```

Strict validation with external tools:

```bash
python zx_normalization_sensitivity.py --large --pairwise-equivalence --strict-qcec
```

Variants failing the configured equivalence guard are quarantined and excluded from sensitivity statistics.

---

## Random seeds and reproducibility

The randomized generators support explicit seed control.

Single seed:

```bash
python zx_normalization_sensitivity.py --medium --random-seeds 0
```

Multiple seeds:

```bash
python zx_normalization_sensitivity.py --large --random-seeds 0 1 2 3 4
```

The seed list is recorded in the generated metadata files.

---

## External OpenQASM benchmarks

The pipeline can import OpenQASM 2 benchmark directories.

```bash
python zx_normalization_sensitivity.py \
    --medium \
    --qasm-dir ./benchmarks_qasm
```

Imported circuits are analyzed together with generated benchmark families.

---

## Output structure

Typical generated directories:

```text
artifacts_normsens/
├── csv/
├── json/
├── graphml/
├── latex/
├── plots/
├── counterexamples/
├── quarantine/
└── summary_artifacts.zip
```

### Important artifact groups

#### CSV summaries

```text
variant_metrics.csv
circuit_summary.csv
family_summary.csv
strategy_health.csv
```

#### JSON artifacts

```text
conclusion.json
family_statistics.json
replay_gap_summary.json
```

#### Graph exports

```text
*.graphml
```

#### LaTeX/TikZ exports

```text
*.tex
```

---

## Recommended repository layout

Suggested GitHub structure:

```text
repository/
├── README.md
├── requirements.txt
├── zx_normalization_sensitivity.py
├── LICENSE
├── examples/
├── benchmark_qasm/
├── docs/
└── results/
```

Large generated experiment artifacts are usually better distributed through GitHub Releases or Zenodo archives instead of directly storing them in the repository.

---

## Suggested GitHub topics

Suggested repository tags:

```text
quantum-computing
zx-calculus
sat-solving
clifford-t
t-depth
quantum-circuits
graph-theory
formal-methods
qiskit
pyzx
```
