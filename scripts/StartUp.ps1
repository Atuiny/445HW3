param(
  [string]$TarPath = "./champion-inference-api.tar",
  [int]$HostPort = 8000,
  [string]$ContainerName = "champion-inference-api"
)

$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command not found: '$Name'. Install it and try again."
  }
}

Require-Command docker

if (-not (Test-Path $TarPath)) {
  throw "Tar file not found: $TarPath"
}

Write-Host "Loading Docker image from tar: $TarPath" -ForegroundColor Cyan
$loadOutput = docker load -i $TarPath 2>&1
$loadText = ($loadOutput | Out-String).Trim()
Write-Host $loadText

$imageRef = $null
$match = [regex]::Match($loadText, "Loaded image:\s*(.+)")
if ($match.Success) {
  $imageRef = $match.Groups[1].Value.Trim()
}

Write-Host "\nCheck available images:" -ForegroundColor Cyan
Write-Host "docker images" -ForegroundColor Gray
docker images

if (-not $imageRef) {
  Write-Host "\nCould not auto-detect image name from 'docker load' output." -ForegroundColor Yellow
  Write-Host "Re-run with the image ref you see in 'docker images', e.g.:" -ForegroundColor Yellow
  Write-Host "docker run --rm -p ${HostPort}:8000 <IMAGE_NAME:TAG>" -ForegroundColor Yellow
  return
}

Write-Host "\nRunning container '$ContainerName' from image '$imageRef'..." -ForegroundColor Cyan
try { docker rm -f $ContainerName | Out-Null } catch {}
docker run --rm --name $ContainerName -p "${HostPort}:8000" $imageRef
