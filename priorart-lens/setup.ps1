# One-time setup on Windows. Run in the VS Code terminal:  .\setup.ps1
$ErrorActionPreference = "Stop"
# VS Code config ships in vscode-config/ (copied into .vscode/ here)
if (Test-Path vscode-config) { New-Item -ItemType Directory -Force .vscode | Out-Null; Copy-Item vscode-config\*.json .vscode\ -Force }
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m synth.generate --n 30 --out data/synth
.\.venv\Scripts\python.exe -m synth.generate --n 30 --out data/synth_cam --camera
.\.venv\Scripts\python.exe -m km.synth --n 20 --out data/km
.\.venv\Scripts\python.exe -m km.synth --n 20 --out data/km_cam --camera
.\.venv\Scripts\python.exe -m pytest -q
Write-Host ""
Write-Host "Setup done. Start the app with:  .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
