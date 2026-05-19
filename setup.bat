@echo off
:: setup.bat — Windows bootstrap for codon-rag
:: Creates a virtualenv, installs all dependencies, and prints next steps.
:: Run once from the repo root:  .\setup.bat

setlocal

set VENV_DIR=.venv

echo.
echo [1/4] Creating virtual environment in %VENV_DIR% ...
python -m venv %VENV_DIR%
if errorlevel 1 (
    echo ERROR: Failed to create venv. Make sure Python 3.12+ is on PATH.
    exit /b 1
)

echo.
echo [2/4] Installing core dependencies (requirements.txt) ...
%VENV_DIR%\Scripts\pip install --upgrade pip -q
%VENV_DIR%\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Core dependency install failed.
    exit /b 1
)

echo.
echo [3/4] Installing workbench dependencies (workbench\requirements.txt) ...
%VENV_DIR%\Scripts\pip install -r workbench\requirements.txt
if errorlevel 1 (
    echo ERROR: Workbench dependency install failed.
    exit /b 1
)

echo.
echo [4/4] Done!
echo.
echo ── Activate the environment ────────────────────────────────────────────
echo    .venv\Scripts\activate
echo.
echo ── Then run either UI ──────────────────────────────────────────────────
echo    streamlit run app.py                          ^(port 8501^)
echo    uvicorn workbench.main:app --port 8000 --reload  ^(port 8000^)
echo.
echo ── Or use the CLI ──────────────────────────────────────────────────────
echo    python src\ingest.py --kb codon
echo    python src\rag.py --kb codon "your question"
echo.

endlocal
