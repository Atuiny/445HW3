param(
  [string]$TarPath = "./champion-inference-api.tar",
  [switch]$DownloadFromGitHub,
  [string]$Repo = "Atuiny/445HW3",
  [string]$WorkflowFile = "ml_pipeline.yaml",
  [string]$Branch = "master",
  [string]$ArtifactName = "champion-inference-api-tar",
  [string]$RunId = "",
  [int]$HostPort = 8000,
  [string]$ContainerName = "champion-inference-api"
)

$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command not found: '$Name'. Install it and try again."
  }
}

if ($DownloadFromGitHub) {
  Require-Command gh

  Write-Host "Checking GitHub CLI auth..." -ForegroundColor Cyan
  try {
    gh auth status | Out-Null
  } catch {
    throw "Not logged into GitHub CLI. Run: gh auth login"
  }

  if (-not $RunId) {
    Write-Host "Finding latest successful workflow run on $Repo ($WorkflowFile @ $Branch)..." -ForegroundColor Cyan
    $RunId = gh run list --repo $Repo --workflow $WorkflowFile --branch $Branch --status success --limit 1 --json databaseId --jq '.[0].databaseId'
  }

  if (-not $RunId) {
    throw "No successful runs found yet for $WorkflowFile on branch $Branch. Run the workflow once in GitHub Actions first."
  }

  $artifactDir = Join-Path $PWD "ci_artifacts"
  if (Test-Path $artifactDir) {
    Remove-Item -Recurse -Force $artifactDir
  }
  New-Item -ItemType Directory -Path $artifactDir | Out-Null

  Write-Host "Downloading artifact '$ArtifactName' from run $RunId..." -ForegroundColor Cyan
  $null = gh run download $RunId --repo $Repo --name $ArtifactName -D $artifactDir

  $tar = Get-ChildItem -Path $artifactDir -Filter "*.tar" -Recurse | Select-Object -First 1
  if (-not $tar) {
    throw "Download succeeded but no .tar file was found under: $artifactDir"
  }

  $TarPath = $tar.FullName
  Write-Host "Downloaded tar: $TarPath" -ForegroundColor Green
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
