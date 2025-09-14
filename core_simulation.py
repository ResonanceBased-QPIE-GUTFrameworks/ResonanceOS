#!/usr/bin/env python3
"""
core_simulation.py

Scrubbed, public-safe core module for running baseline vs. alternative quantum
circuits, performing parameter sweeps, and exporting clean results.
"""

from __future__ import annotations
import json
import os
import time
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, state_fidelity

try:
    from qiskit import IBMQ
    _HAS_IBMQ = True
except Exception:
    _HAS_IBMQ = False


def baseline_circuit(n_qubits: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def alt_circuit(n_qubits: int = 3, param1: float = 0.1, param2: float = 4.0) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        theta = (q + 1) / float(param2) * (np.pi / 4.0)
        qc.ry(2.0 * theta, q)
        qc.rz(param1 * (q + 1), q)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def get_backend(backend_name: str | None = None, use_simulator_fallback: bool = True):
    if backend_name is None:
        return Aer.get_backend('qasm_simulator')
    name = backend_name.lower()
    if name in ('qasm_simulator', 'aer_simulator', 'sim'):
        return Aer.get_backend('qasm_simulator')
    if (name.startswith('ibm') or name.startswith('ibmq')) and _HAS_IBMQ:
        token = os.environ.get("IBMQ_TOKEN", "").strip()
        try:
            if token:
                IBMQ.save_account(token, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.providers()[0]
            return provider.get_backend(backend_name)
        except Exception as e:
            if use_simulator_fallback:
                print(f"[WARN] Could not resolve IBM backend '{backend_name}' ({e}). Using local simulator.")
                return Aer.get_backend('qasm_simulator')
            raise
    else:
        if use_simulator_fallback:
            print(f"[WARN] Unknown backend '{backend_name}'. Using local simulator instead.")
            return Aer.get_backend('qasm_simulator')
        raise ValueError(f"Unknown backend '{backend_name}' and simulator fallback disabled.")


def run_counts(qc: QuantumCircuit, backend, shots: int = 1024) -> Dict[str, int]:
    job = execute(qc, backend=backend, shots=shots)
    res = job.result()
    return res.get_counts()


def estimate_fidelity_to_simple_ghz_like(qc_no_meas: QuantumCircuit) -> float:
    sv_backend = Aer.get_backend('statevector_simulator')
    state = execute(qc_no_meas, backend=sv_backend).result().get_statevector()
    n = qc_no_meas.num_qubits
    target = Statevector.from_label('0' * n) + Statevector.from_label('1' * n)
    target = target / np.sqrt(2.0)
    return float(state_fidelity(state, target))


def sweep_parameters(n_qubits: int,
                     param1_values: List[float],
                     param2_values: List[float],
                     trials: int,
                     shots: int,
                     backend_name: str | None = None) -> List[Dict[str, Any]]:
    backend = get_backend(backend_name)
    results_rows: List[Dict[str, Any]] = []
    for p1 in param1_values:
        for p2 in param2_values:
            for trial in range(trials):
                base_qc = baseline_circuit(n_qubits)
                alt_qc = alt_circuit(n_qubits, param1=p1, param2=p2)
                base_counts = run_counts(base_qc, backend, shots=shots)
                alt_counts = run_counts(alt_qc, backend, shots=shots)
                alt_no_meas = alt_qc.remove_final_measurements(inplace=False)
                fid = estimate_fidelity_to_simple_ghz_like(alt_no_meas)
                results_rows.append({
                    "timestamp": int(time.time()),
                    "n_qubits": n_qubits,
                    "param1": p1,
                    "param2": p2,
                    "trial": trial,
                    "shots": shots,
                    "backend": backend_name or "qasm_simulator",
                    "baseline_counts": base_counts,
                    "alternative_counts": alt_counts,
                    "fidelity_vs_reference": fid
                })
    return results_rows


def save_json(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _parse_args():
    p = argparse.ArgumentParser(description="Scrubbed parameter sweep runner")
    p.add_argument("--n-qubits", type=int, default=3)
    p.add_argument("--param1", type=str, default="0.05,0.10,0.15")
    p.add_argument("--param2", type=str, default="3,4,5")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--out-json", type=str, default="results/results_sweep.json")
    p.add_argument("--out-csv", type=str, default="results/results_sweep.csv")
    return p.parse_args()


def main():
    args = _parse_args()
    param1_values = [float(x.strip()) for x in args.param1.split(",") if x.strip()]
    param2_values = [float(x.strip()) for x in args.param2.split(",") if x.strip()]
    rows = sweep_parameters(
        n_qubits=args.n_qubits,
        param1_values=param1_values,
        param2_values=param2_values,
        trials=args.trials,
        shots=args.shots,
        backend_name=args.backend
    )
    save_json(rows, args.out_json)
    save_csv(rows, args.out_csv)
    print(f"[OK] Wrote {len(rows)} rows → {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
