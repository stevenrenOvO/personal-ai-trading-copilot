"""Exercise launcher environment inheritance without starting a real process."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


NAMES = ("PYTHONIOENCODING", "COPILOT_DEMO_MODE", "COPILOT_DB_PATH", "COPILOT_FIXTURE_DIR")


def _run_launcher(tmp_path, *, preset, fail=False, modes=("demo",)):
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        pytest.skip("PowerShell is required for Windows launcher regression")
    root = tmp_path / "unpacked-project"
    root.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "start.ps1", root / "start.ps1")
    (root / "VERSION").write_text("1.1\n", encoding="utf-8")
    fake_python = root / ".venv" / "Scripts" / "python.exe"
    fake_python.parent.mkdir(parents=True)
    fake_python.touch()
    configuration = root / "harness.json"
    config = {"root": str(root), "preset": preset, "fail": fail, "modes": list(modes)}
    configuration.write_text(json.dumps(config), encoding="utf-8")
    # Every external interaction is mocked. The script under test is unchanged
    # except that it lives in an empty temporary project, away from real state.
    code = r'''
$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath $env:LAUNCHER_TEST_CONFIG -Raw | ConvertFrom-Json
$names = @('PYTHONIOENCODING', 'COPILOT_DEMO_MODE', 'COPILOT_DB_PATH', 'COPILOT_FIXTURE_DIR')
foreach ($name in $names) {
    $value = if ($config.preset) { 'caller-' + $name } else { $null }
    [Environment]::SetEnvironmentVariable($name, $value, 'Process')
}
function Snapshot-Environment {
    $snapshot = @{}
    foreach ($name in $names) { $snapshot[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    return $snapshot
}
$initial = Snapshot-Environment
$global:captures = @()
$global:outcomes = @()
$global:started = $false
$global:launchPort = 0
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle, $RedirectStandardOutput, $RedirectStandardError, [switch]$PassThru)
    $global:captures += ,@{ environment = (Snapshot-Environment); path = $FilePath; arguments = $ArgumentList }
    if ($config.fail) { throw 'mock launch failure' }
    $global:launchPort = [int]$ArgumentList[-1]
    $global:started = $true
    return [pscustomobject]@{ Id = 42420; HasExited = $false; Path = $FilePath; StartTime = [datetime]'2026-10-24T12:00:00Z' }
}
function Invoke-WebRequest {
    param($Uri, $TimeoutSec)
    if (-not $global:started) { throw 'mock no listener' }
    return [pscustomobject]@{ StatusCode = 200; Content = '<title>Personal AI Trading Copilot</title>' }
}
function Invoke-RestMethod {
    param($Uri, $TimeoutSec)
    return [pscustomobject]@{ status = 'ok'; version = '1.1' }
}
function Start-Sleep { param($Milliseconds) }
function netstat { ' TCP 127.0.0.1:' + $global:launchPort + ' 0.0.0.0:0 LISTENING 42421' }
function Get-Process {
    param($Id)
    return [pscustomobject]@{ Id = $Id; Path = (Join-Path $config.root '.venv\Scripts\python.exe'); StartTime = [datetime]'2026-10-24T12:00:00Z' }
}
function Set-Content { param($LiteralPath, $Value, $Encoding, [Parameter(ValueFromPipeline=$true)]$InputObject) process {} }
foreach ($mode in $config.modes) {
    $global:started = $false
    $caught = $null
    try {
        if ($mode -eq 'demo') { & (Join-Path $config.root 'start.ps1') -Demo }
        else { & (Join-Path $config.root 'start.ps1') }
    } catch { $caught = $_.Exception.Message }
    $global:outcomes += ,@{ mode = $mode; error = $caught; after = (Snapshot-Environment) }
}
@{ initial = $initial; captures = @($global:captures); outcomes = @($global:outcomes) } | ConvertTo-Json -Depth 8 -Compress
'''
    encoded = base64.b64encode(code.encode("utf-16le")).decode("ascii")
    env = os.environ.copy()
    env["LAUNCHER_TEST_CONFIG"] = str(configuration)
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                            env=env, cwd=root, capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
    # Write-Host emits the launch message before the harness JSON.
    output = next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{"'))
    assert not list(root.glob("*.pid")) and not list(root.glob("*.log"))
    assert not (root / "data").exists()
    return json.loads(output), root


@pytest.mark.parametrize("preset", [False, True], ids=["unset-caller", "custom-caller"])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "launch-failure"])
@pytest.mark.parametrize("mode", ["demo", "production"])
def test_launch_environment_is_temporary_for_success_and_failure(tmp_path, preset, fail, mode):
    observed, root = _run_launcher(tmp_path, preset=preset, fail=fail, modes=(mode,))
    assert len(observed["captures"]) == 1
    child = observed["captures"][0]["environment"]
    initial = observed["initial"]
    assert child["PYTHONIOENCODING"] == "utf-8"
    assert child["COPILOT_DEMO_MODE"] == ("1" if mode == "demo" else "0")
    if mode == "demo":
        assert Path(child["COPILOT_DB_PATH"]) == root / "data" / "demo" / "market.db"
        assert Path(child["COPILOT_FIXTURE_DIR"]) == root / "tests" / "fixtures"
    else:
        assert child["COPILOT_DB_PATH"] == initial["COPILOT_DB_PATH"]
        assert child["COPILOT_FIXTURE_DIR"] == initial["COPILOT_FIXTURE_DIR"]
    outcome = observed["outcomes"][0]
    assert outcome["after"] == initial
    assert outcome["error"] == ("mock launch failure" if fail else None)


def test_demo_then_production_in_same_terminal_uses_original_account_directory(tmp_path):
    observed, _ = _run_launcher(tmp_path, preset=True, modes=("demo", "production"))
    assert len(observed["captures"]) == 2
    assert all(outcome["error"] is None for outcome in observed["outcomes"])
    assert all(outcome["after"] == observed["initial"] for outcome in observed["outcomes"])
    demo, production = [capture["environment"] for capture in observed["captures"]]
    assert demo["COPILOT_DEMO_MODE"] == "1" and production["COPILOT_DEMO_MODE"] == "0"
    assert production["COPILOT_DB_PATH"] == observed["initial"]["COPILOT_DB_PATH"]
    assert production["COPILOT_DB_PATH"] != demo["COPILOT_DB_PATH"]
