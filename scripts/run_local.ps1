# Run the dashboard locally against data/app/*.parquet (download with the export task first).
param([int]$Port = 8501)
$root = Split-Path -Parent $PSScriptRoot
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
if (-not $env:DATABRICKS_CONFIG_PROFILE) { $env:DATABRICKS_CONFIG_PROFILE = "DEFAULT" }
$env:HEAT_ARTIFACTS = "$root/data/app" -replace '\\','/'
if (-not $env:DATABRICKS_WAREHOUSE_ID) { Write-Warning "DATABRICKS_WAREHOUSE_ID is not set: the planner agent's SQL tools won't work" }
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
& powershell -File "$root/scripts/sync_app.ps1" | Out-Null
Set-Location "$root/app"
& "$root/.venv/Scripts/streamlit.exe" run app.py --server.port $Port --server.headless true
