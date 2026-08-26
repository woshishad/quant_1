param(
  [string]$PythonExe = "D:\conda-envs\quant-competition-sim\python.exe",
  [string]$SourceDir = ".\results\blend_final_120k_global_shrink_recency_weighted_8blocks",
  [string]$FinalDir = ".\results\final_recommended_submission",
  [string]$CandidateRanking = ".\results\final_submission_candidate_ranking.csv",
  [string]$SampleSubmission = ".\data\raw\public_release_20260630\public_release_20260630\data\sample_submission.csv"
)

$ErrorActionPreference = "Stop"

Write-Host "== Package final recommended submission ==" -ForegroundColor Cyan
Write-Host "Python: $PythonExe"
Write-Host "SourceDir: $SourceDir"
Write-Host "FinalDir: $FinalDir"

if (-not (Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

& $PythonExe -u .\code\experiments\asset_all_tcn\package_final_submission.py `
  --source-dir $SourceDir `
  --final-dir $FinalDir `
  --candidate-ranking $CandidateRanking `
  --sample-submission $SampleSubmission

Write-Host "Package finished. Recommended upload file:" -ForegroundColor Green
Write-Host "$FinalDir\submission.zip" -ForegroundColor Green
