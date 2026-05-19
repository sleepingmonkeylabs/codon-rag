#!/usr/bin/env bash
# setup.sh — macOS / Linux bootstrap for codon-rag
# Creates a virtualenv, installs all dependencies, and prints next steps.
# Run once from the repo root:  bash setup.sh

set -euo pipefail

VENV_DIR=".venv"

echo
echo "[1/4] Creating virtual environment in ${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}"

echo
echo "[2/4] Installing core dependencies (requirements.txt) ..."
"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install -r requirements.txt

echo
echo "[3/4] Installing workbench dependencies (workbench/requirements.txt) ..."
"${VENV_DIR}/bin/pip" install -r workbench/requirements.txt

echo
echo "[4/4] Done!"
echo
echo "── Activate the environment ────────────────────────────────────────────"
echo "   source .venv/bin/activate"
echo
echo "── Then run either UI ──────────────────────────────────────────────────"
echo "   streamlit run app.py                               # port 8501"
echo "   uvicorn workbench.main:app --port 8000 --reload    # port 8000"
echo
echo "── Or use the CLI ──────────────────────────────────────────────────────"
echo "   python src/ingest.py --kb codon"
echo "   python src/rag.py --kb codon 'your question'"
echo
