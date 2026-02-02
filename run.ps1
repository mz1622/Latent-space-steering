# run_pope_llava_steering.ps1
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\run_pope_llava_steering.ps1
# Or:
#   .\run_pope_llava_steering.ps1

$ErrorActionPreference = "Stop"

$steeringPath = "outputs\20260203_020758\vti_baseline\artifacts"

Write-Host "== Checking steering path: $steeringPath"
if (-not (Test-Path $steeringPath)) {
  throw "Steering path not found: $steeringPath"
}

Write-Host "== Running evaluation..."
python scripts/evaluate_multi_model.py `
  --model-type llava `
  --benchmark pope `
  --pope-type adversarial `
  --use-steering `
  --steering-path "$steeringPath" `
  --alpha-image 0.2 `
  --alpha-text 0.4 `
