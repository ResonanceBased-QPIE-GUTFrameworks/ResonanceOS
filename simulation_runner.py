#!/usr/bin/env python3
"""
simulation_runner.py

Colab/CLI-friendly runner that imports the scrubbed core module,
executes a parameter sweep with sensible defaults, and saves JSON/CSV.
"""

from __future__ import annotations
from core_simulation import main as core_main

if __name__ == "__main__":
    core_main()
