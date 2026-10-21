# One-time setup: creates .venv, installs dependencies, checks tools. Run from the repo root:  ./setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
$ver = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "Python $ver"
if ([version]$ver -lt [version]"3.10") { throw "Python 3.10+ required (found $ver)" }

if (-not (Test-Path .venv)) { & $py -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# Face detector model (YuNet, ~230 KB). Note: github.com/.../raw/ returns a Git LFS pointer, so use the media URL.
New-Item -ItemType Directory -Force models | Out-Null
$face = "models\face_detection_yunet_2023mar.onnx"
if (-not (Test-Path $face) -or (Get-Item $face).Length -lt 100000) {
  Invoke-WebRequest "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile $face
}
Write-Host "[ok]      face model ($((Get-Item $face).Length) bytes)"

# Whisper model for the no-VTT path (DECISIONS D11, ~57 MB)
$wm = "models\ggml-base-q5_1.bin"
if (-not (Test-Path $wm)) {
  Invoke-WebRequest "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin" -OutFile $wm
}
Write-Host "[ok]      whisper model ($([math]::Round((Get-Item $wm).Length / 1MB)) MB)"

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env from .env.example - add your DeepSeek key" }

# Tool checks (paths from .env)
$envVars = @{}
Get-Content .env | Where-Object { $_ -match '^\s*([A-Z_]+)\s*=\s*(.*)$' } | ForEach-Object {
  $m = [regex]::Match($_, '^\s*([A-Z_]+)\s*=\s*(.*)$'); $envVars[$m.Groups[1].Value] = $m.Groups[2].Value.Trim('"')
}
foreach ($k in "FFMPEG_BIN","FFPROBE_BIN","WHISPER_CLI","WHISPER_MODEL","DATA_DIR") {
  $p = $envVars[$k]
  if ($p -and (Test-Path $p)) { Write-Host "[ok]      $k" } else { Write-Host "[missing] $k -> $p" -ForegroundColor Yellow }
}
if ($envVars["DEEPSEEK_API_KEY"] -like "your-*") { Write-Host "[todo]    DEEPSEEK_API_KEY not set in .env" -ForegroundColor Yellow }

if (-not (Test-Path .git)) { git init | Out-Null; Write-Host "git repo initialised" }
Write-Host "Setup done. Activate with: .\.venv\Scripts\Activate.ps1"
