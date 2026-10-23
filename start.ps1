param([int]$Port = 8000, [switch]$OpenBrowser, [switch]$Demo)
$ErrorActionPreference = 'Stop'
if ($Demo -and -not $PSBoundParameters.ContainsKey('Port')) { $Port = 8001 }
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$expectedVersion = (Get-Content -LiteralPath (Join-Path $projectRoot 'VERSION') -Raw).Trim()
$recordName = if ($Demo) { 'server-demo' } else { 'server' }
$url = "http://127.0.0.1:$Port/"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Python environment missing. Create .venv and install requirements.txt first.'
}
try {
    $page = Invoke-WebRequest -Uri $url -TimeoutSec 3
    if ($page.StatusCode -eq 200 -and $page.Content.Contains('<title>Personal AI Trading Copilot</title>')) {
        $runningHealth = Invoke-RestMethod -Uri ($url + 'health') -TimeoutSec 3
        if ($runningHealth.version -ne $expectedVersion) { throw "Version mismatch: running $($runningHealth.version), expected $expectedVersion. Stop the existing service first." }
        $expectedMode = if ($Demo) { 'demo' } else { 'production' }
        if ($runningHealth.mode -ne $expectedMode) { throw "Mode mismatch on port $Port. Stop that service or select another port." }
        Write-Host "Already running: $url"
        if ($OpenBrowser) { Start-Process $url }
        exit 0
    }
    throw "Port $Port is occupied by a different service."
} catch {
    if ($_.Exception.Message -like '*occupied*' -or $_.Exception.Message -like '*Version mismatch*' -or $_.Exception.Message -like '*Mode mismatch*') { throw }
}
$stdout = Join-Path $projectRoot "$recordName.out.log"
$stderr = Join-Path $projectRoot "$recordName.err.log"
foreach ($logPath in @($stdout, $stderr)) {
    if ((Test-Path -LiteralPath $logPath) -and (Get-Item -LiteralPath $logPath).Length -gt 5MB) {
        $logBackup = Join-Path $projectRoot 'data\tmp\logs'
        New-Item -ItemType Directory -Path $logBackup -Force | Out-Null
        Move-Item -LiteralPath $logPath -Destination (Join-Path $logBackup ((Split-Path $logPath -Leaf) + '.' + [DateTime]::UtcNow.ToString('yyyyMMddHHmmssfff')))
    }
}
$savedEnvironment = @{}
$launchEnvironmentNames = @('PYTHONIOENCODING', 'COPILOT_DEMO_MODE', 'COPILOT_DB_PATH', 'COPILOT_FIXTURE_DIR')
foreach ($name in $launchEnvironmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $env:PYTHONIOENCODING = 'utf-8'
    if ($Demo) {
        $env:COPILOT_DEMO_MODE = '1'
        $env:COPILOT_DB_PATH = Join-Path $projectRoot 'data\demo\market.db'
        $env:COPILOT_FIXTURE_DIR = Join-Path $projectRoot 'tests\fixtures'
    } else {
        $env:COPILOT_DEMO_MODE = '0'
    }
    $process = Start-Process -FilePath $pythonPath -ArgumentList '-u','-m','backend.api.app','--port',"$Port" -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
} finally {
    foreach ($name in $launchEnvironmentNames) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
Set-Content -LiteralPath (Join-Path $projectRoot "$recordName.pid") -Value $process.Id -Encoding ascii
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 500
    if ($process.HasExited) {
        throw "Service exited. Inspect $stderr"
    }
    try {
        $health = Invoke-RestMethod -Uri ($url + 'health') -TimeoutSec 1
        $page = Invoke-WebRequest -Uri $url -TimeoutSec 1
        if ($health.status -eq 'ok' -and $page.Content.Contains('<title>Personal AI Trading Copilot</title>')) {
            $listener = netstat -ano -p TCP | Where-Object { $_ -match "127\.0\.0\.1:$Port\s+.*LISTENING\s+\d+$" } | Select-Object -First 1
            if (-not $listener) { throw 'Cannot identify service listener.' }
            $workerId = [int](($listener.Trim() -split '\s+')[-1])
            $worker = Get-Process -Id $workerId
            $processInfo = @{
                root = $projectRoot
                launcher = @{ id = $process.Id; path = $process.Path; ticks = $process.StartTime.ToUniversalTime().Ticks.ToString() }
                worker = @{ id = $worker.Id; path = $worker.Path; ticks = $worker.StartTime.ToUniversalTime().Ticks.ToString() }
            }
            $processInfo | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $projectRoot "$recordName.process.json") -Encoding utf8
            Write-Host "Started: $url"
            if ($OpenBrowser) { Start-Process $url }
            exit 0
        }
    } catch { }
}
throw "Service did not become available. Inspect $stderr"
