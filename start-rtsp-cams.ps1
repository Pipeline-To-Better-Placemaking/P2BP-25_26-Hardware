# start-rtsp-cams.ps1
$ErrorActionPreference = 'Stop'
Write-Host "=== RTSP multi-cam from files (MediaMTX + FFmpeg) ===`n"

function Require-Command($name, $hint) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    Write-Host "Missing: $name"
    Write-Host "Hint: $hint"
    throw "Required command not found: $name"
  }
}

function Start-MediaMTX {
  $exe = Join-Path $PSScriptRoot "mediamtx.exe"
  $cfg = Join-Path $PSScriptRoot "mediamtx.yml"

    if (Test-Path $exe) {
        Write-Host "`nStarting local mediamtx.exe on rtsp://127.0.0.1:8554 ..."
    $proc = Start-Process -FilePath $exe -ArgumentList "`"$cfg`"" -WindowStyle Hidden -PassThru
        Start-Sleep -Seconds 1
        try {
            $ok = Test-NetConnection 127.0.0.1 -Port 8554 -WarningAction SilentlyContinue
            if (-not $ok.TcpTestSucceeded) {
                Write-Warning "mediamtx.exe started but port 8554 not open yet. Waiting a bit more..."
                Start-Sleep -Seconds 2
            }
        } catch { }

        return "exe"
    }

    if (Get-Command docker -ErrorAction SilentlyContinue) {
        try {
            Write-Host "`nStarting MediaMTX via Docker on rtsp://127.0.0.1:8554 ..."
            docker rm -f mediamtx | Out-Null 2>$null
            docker run -d --name mediamtx -p 8554:8554 bluenviron/mediamtx:latest | Out-Null
            Start-Sleep -Seconds 1
            return "docker"
        } catch {
            Write-Warning "Docker start failed: $($_.Exception.Message)"
        }
    }

    throw "Could not start MediaMTX. Place mediamtx.exe beside the script or start Docker Desktop."
}

function Wait-RTSPReady {
  param([int]$Port = 8554, [int]$TimeoutSec = 10)
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    try {
      $t = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue
      if ($t.TcpTestSucceeded) { return $true }
    } catch { }
    Start-Sleep -Milliseconds 400
  }
  return $false
}


function Ensure-FFmpeg {
  if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { return }

  Write-Host "FFmpeg not found on PATH."
  $auto = Read-Host "Attempt to install FFmpeg automatically? [Y/n]"
  if ($auto -match '^(n|no)$') {
    Write-Host "Please install FFmpeg and re-run:"
    Write-Host "  winget install FFmpeg.FFmpeg"
    Write-Host "  # or: winget install Gyan.FFmpeg"
    Write-Host "  # or: choco install ffmpeg"
    Write-Host "  # or: scoop install ffmpeg"
    throw "FFmpeg is required."
  }

  $installed = $false

  if (Get-Command winget -ErrorAction SilentlyContinue) {
    try {
      winget install -e --id FFmpeg.FFmpeg --accept-source-agreements --accept-package-agreements | Out-Null
      if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { $installed = $true }
      if (-not $installed) {
        winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements | Out-Null
        if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { $installed = $true }
      }
    } catch { }
  }

  if (-not $installed -and (Get-Command choco -ErrorAction SilentlyContinue)) {
    try {
      choco install -y ffmpeg | Out-Null
      if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { $installed = $true }
    } catch { }
  }

  if (-not $installed -and (Get-Command scoop -ErrorAction SilentlyContinue)) {
    try {
      scoop install ffmpeg | Out-Null
      if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { $installed = $true }
    } catch { }
  }

  if (-not $installed) {
    Write-Host "Automatic install failed or no package manager found."
    Write-Host "Manual options:"
    Write-Host "  winget install FFmpeg.FFmpeg   # or Gyan.FFmpeg"
    Write-Host "  choco install ffmpeg"
    Write-Host "  scoop install ffmpeg"
    throw "FFmpeg is required."
  } else {
    Write-Host "FFmpeg installed. If 'ffmpeg' is still not recognized, open a new PowerShell window to refresh PATH."
  }
}


# ---- Prereqs ----
Ensure-FFmpeg

# ---- Basic prompts ----
$N   = Read-Host "How many cameras? [3]"; if ([string]::IsNullOrWhiteSpace($N)) { $N = 3 } else { $N = [int]$N }
$RES = Read-Host "Output resolution WxH [1280x720]"; if ([string]::IsNullOrWhiteSpace($RES)) { $RES = "1280x720" }
$FPS = Read-Host "Output FPS [30]"; if ([string]::IsNullOrWhiteSpace($FPS)) { $FPS = 30 } else { $FPS = [int]$FPS }
$VIEW = Read-Host "Launch viewer after streams are running? [Y]"; if ([string]::IsNullOrWhiteSpace($VIEW)) { $VIEW = "Y" }

# ---- Start MediaMTX server ----
$mediaMtxMode = Start-MediaMTX
if (-not (Wait-RTSPReady -Port 8554 -TimeoutSec 20)) {
  Write-Warning "RTSP port 8554 not open yet; waiting 5 more seconds..."
  Start-Sleep -Seconds 5
}

# ---- Collect inputs ----
$files = @()
$offs  = @()
for ($i = 0; $i -lt $N; $i++) {
  $default = "4p-c$($i).avi"
  $f = Read-Host "Path for cam$i file [$default]"
  if ([string]::IsNullOrWhiteSpace($f)) { $f = $default }

  if (-not (Test-Path -LiteralPath $f)) {
    Write-Host "  File not found. Enter a valid path (or Ctrl+C to abort)."
    $f = Read-Host "  Path for cam$i file"
  }
  $rp = Resolve-Path -LiteralPath $f -ErrorAction SilentlyContinue
  if ($rp) { $f = $rp.Path }
  $files += $f

  $o = Read-Host "Offset (seconds) for cam$i [0]"
  if ([string]::IsNullOrWhiteSpace($o)) { $o = 0 }
  $offs += [double]$o
}

# ---- Launch single publisher ----
$logDir   = Join-Path $env:TEMP "rtsp_cam_logs"
$pidsPath = Join-Path $env:TEMP "rtsp_ffmpeg_pids.txt"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Set-Content -Path $pidsPath -Value ""

$log = Join-Path $logDir "mux.log"
$vcodec = "libx264"

# ---- Build inputs ----
$inList = @()
for ($i = 0; $i -lt $N; $i++) {
  Write-Host "Muxing cam${i}: $($files[$i]) (offset $($offs[$i])s) -> rtsp://127.0.0.1:8554/cam$($i)"
  $inList += @(
    '-re','-stream_loop','-1',
    '-itsoffset', $offs[$i].ToString(),
    '-i', "`"$($files[$i])`""
  )
}

# ---- Build filter graph ----
$chains = @()
for ($i = 0; $i -lt $N; $i++) {
  $chains += "[${i}:v]scale=$RES,fps=$FPS,format=yuv420p[v$i]"
}
$filter = ($chains -join ';')

# ---- Build outputs ----
$outList = @()
for ($i = 0; $i -lt $N; $i++) {
  $url = "rtsp://127.0.0.1:8554/cam$($i)"
  $outList += @(
    '-map', "[v$i]",
    '-c:v', $vcodec, '-preset','veryfast','-tune','zerolatency',
    '-g', $FPS.ToString(),
    '-an',
    '-f','rtsp','-rtsp_transport','tcp', $url
  )
}

# ---- Compose final arg string ----
$argList   = @('-hide_banner','-loglevel','info','-nostdin') + $inList + @('-filter_complex', "`"$filter`"") + $outList
$argString = $argList -join ' '

$proc = Start-Process -FilePath "ffmpeg" `
          -ArgumentList $argString `
          -RedirectStandardError $log `
          -NoNewWindow `
          -PassThru

Add-Content -Path $pidsPath -Value $proc.Id

Write-Host "Launched single ffmpeg muxer -> $N RTSP outputs"
Write-Host "Log: $log"


# ---- Print viewing/help info ----
function Get-ViewerCmd([string]$u) {
  if (Get-Command ffplay -ErrorAction SilentlyContinue) { return "ffplay -rtsp_transport tcp $u" }
  elseif (Get-Command mpv -ErrorAction SilentlyContinue) { return "mpv --rtsp-transport=tcp $u" }
  elseif (Test-Path "C:\Program Files\VideoLAN\VLC\vlc.exe") { return "`"C:\Program Files\VideoLAN\VLC\vlc.exe`" $u" }
  else { return "# Install a viewer: winget install Gyan.FFmpeg (ffplay), or winget install VideoLAN.VLC, or winget install Shinchiro.MPV" }
}

$urls = @()
Write-Host "`n== Streams (copy/paste to view) =="
for ($i = 0; $i -lt $N; $i++) {
  $u = "rtsp://127.0.0.1:8554/cam$($i)"
  $urls += $u
  Write-Host "  $(Get-ViewerCmd $u)"
}

if ($VIEW -match '^(y|yes)$') {
  Write-Host "`nLaunching viewers..."
  $launchAt = (Get-Date).AddSeconds(2)  # common start ~2s in the future
  foreach ($u in $urls) {
    $ms = [int][Math]::Max(0, ($launchAt - (Get-Date)).TotalMilliseconds)
    $cmd = "Start-Sleep -Milliseconds $ms; ffplay -rtsp_transport tcp -fflags nobuffer -flags low_delay -framedrop `"$u`""
    Start-Process powershell -ArgumentList "-NoProfile","-Command",$cmd -WindowStyle Hidden
  }
}

Write-Host "`nLogs live at: $logDir"
Write-Host "Tail log:     Get-Content -Wait `"$logDir\mux.log`""
Write-Host "Stop sender:  Get-Content `"$pidsPath`" | ForEach-Object { try { Stop-Process -Id `$_ -Force } catch {} }"
