# Copy the shared heat package into the app folder (Databricks Apps upload only app/).
$root = Split-Path -Parent $PSScriptRoot
$dst = Join-Path $root "app\heat"
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
Copy-Item -Recurse (Join-Path $root "src\heat") $dst
Get-ChildItem -Recurse -Directory $dst -Filter "__pycache__" | Remove-Item -Recurse -Force
Write-Output "synced src/heat -> app/heat"
