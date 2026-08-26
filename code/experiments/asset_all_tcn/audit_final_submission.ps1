param(
  [string]$PythonExe = "D:\conda-envs\quant-competition-sim\python.exe",
  [string]$FinalDir = ".\results\final_recommended_submission",
  [string]$SourceDir = ".\results\blend_final_120k_global_shrink_recency_weighted_8blocks",
  [string]$SampleSubmission = ".\data\raw\public_release_20260630\public_release_20260630\data\sample_submission.csv"
)

$ErrorActionPreference = "Stop"

Write-Host "== Quant competition final submission audit ==" -ForegroundColor Cyan
Write-Host "Python: $PythonExe"
Write-Host "FinalDir: $FinalDir"
Write-Host "SourceDir: $SourceDir"

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

& $PythonExe -c "import torch; print('torch_cuda_available=', torch.cuda.is_available()); print('torch_device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

& $PythonExe -u .\code\experiments\asset_all_tcn\audit_submission_package.py `
  --final-dir $FinalDir `
  --source-dir $SourceDir `
  --sample-submission $SampleSubmission

Write-Host "Audit finished. Recommended upload file:" -ForegroundColor Green
Write-Host "$FinalDir\submission.zip" -ForegroundColor Green
