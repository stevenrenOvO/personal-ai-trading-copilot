param([switch]$Demo)
$ErrorActionPreference = 'Stop'
$recordName = if ($Demo) { 'server-demo' } else { 'server' }
$infoPath = Join-Path $PSScriptRoot "$recordName.process.json"
if (-not (Test-Path -LiteralPath $infoPath)) { throw 'No verified process record. This script only stops services launched by start.cmd.' }
$info = Get-Content -LiteralPath $infoPath -Raw | ConvertFrom-Json
if ($info.root -ne $PSScriptRoot) { throw 'Process record belongs to another project.' }
foreach ($record in @($info.worker, $info.launcher)) {
    $service = Get-Process -Id ([int]$record.id) -ErrorAction SilentlyContinue
    if (-not $service) { continue }
    if ($service.Path -ne $record.path -or $service.StartTime.ToUniversalTime().Ticks.ToString() -ne $record.ticks) {
        throw 'PID was reused or process identity changed. No matching process was stopped.'
    }
    Stop-Process -Id $service.Id
}
Write-Host 'Service stopped.'
