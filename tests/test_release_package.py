"""Source release safety, integrity, reproducibility and clean extraction checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import zipfile

import pytest

from scripts.build_release import (
    MANIFEST_NAME, REQUIRED_FILES, ReleaseError, _is_link, build_release,
    collect_payload, smoke_test_archive, verify_archive,
)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for name in REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1.1\n" if name == "VERSION" else "source content\n", encoding="utf-8")
    return root


def write(root, name, content="MUST_NOT_SHIP"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_release_includes_saved_uncommitted_files_and_excludes_machine_data(source, tmp_path):
    # There is deliberately no Git repository: edits/untracked source are inputs.
    permitted = ["backend/new_feature.py", "scripts/new_check.py", "tests/test_new.py",
                 "tests/fixtures/daily.csv", "docs/new-release.md", "requirements-runtime.txt",
                 "setup.ps1", "start-demo.cmd", "backend/dashboard/static/style.css"]
    for name in permitted:
        write(source, name, "new source")
    forbidden = [".env", ".env.local", ".git/config", ".venv/pyvenv.cfg", "server.pid",
                 "server.process.json", "server.err.log", "data/paper/state.json",
                 "data/journal/records.jsonl", "data/market.db", "data/tmp/credentials.json",
                 "backend/__pycache__/app.pyc", "backend/cache/snapshot.py", "backend/.env",
                 "backend/credentials/token.py", "docs/secrets.md", "docs/.env.notes.md",
                 "tests/fixtures/real-account.csv", "docs/state.json", "tests/runtime.json",
                 "backend/dashboard/static/session.log", "backend/dashboard/static/private.key"]
    for name in forbidden:
        write(source, name)
    result = build_release(source, tmp_path / "out")
    manifest = verify_archive(result.archive)
    names = {entry["path"] for entry in manifest["files"]}
    assert set(permitted).issubset(names)
    assert not set(forbidden) & names
    with zipfile.ZipFile(result.archive) as package:
        assert all(b"MUST_NOT_SHIP" not in package.read(member) for member in package.namelist())


def test_manifest_and_external_checksum_cover_exact_payload(source, tmp_path):
    result = build_release(source, tmp_path / "out")
    manifest = verify_archive(result.archive)
    assert result.checksum_file.read_text().split() == [hashlib.sha256(result.archive.read_bytes()).hexdigest(), result.archive.name]
    with zipfile.ZipFile(result.archive) as package:
        prefix = f"personal-ai-trading-copilot-{result.version}/"
        assert len(package.namelist()) == len(manifest["files"]) + 1
        for item in manifest["files"]:
            content = package.read(prefix + item["path"])
            assert item["size"] == len(content)
            assert item["sha256"] == hashlib.sha256(content).hexdigest()
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in package.infolist())


def test_same_source_bytes_produce_identical_zip_even_after_mtime_changes(source, tmp_path):
    first = build_release(source, tmp_path / "first")
    before = first.archive.read_bytes()
    for path in source.rglob("*"):
        if path.is_file():
            os.utime(path, (1_800_000_000, 1_800_000_000))
    second = build_release(source, tmp_path / "second")
    assert second.archive.read_bytes() == before
    write(source, "backend/untracked.py", "print('saved edit')\n")
    changed = build_release(source, tmp_path / "third")
    assert changed.sha256 != first.sha256


@pytest.mark.parametrize("malicious", [
    "../outside.py", "/absolute.py", "C:/outside.py", "root\\outside.py",
    "personal-ai-trading-copilot-1.1/backend/../../outside.py",
    "personal-ai-trading-copilot-1.1/backend//outside.py",
    "personal-ai-trading-copilot-1.1/backend/./outside.py",
    "personal-ai-trading-copilot-1.1/backend/x.py:stream",
    "personal-ai-trading-copilot-1.1/backend/CON.py",
    "personal-ai-trading-copilot-1.1/backend/x.py.",
])
def test_archive_path_traversal_and_windows_aliases_rejected_before_extract(source, tmp_path, malicious):
    result = build_release(source, tmp_path / "out")
    with zipfile.ZipFile(result.archive, "a") as package:
        package.writestr(malicious, "unsafe")
    with pytest.raises(ReleaseError):
        verify_archive(result.archive)
    assert not (tmp_path / "outside.py").exists()


def test_added_sensitive_file_rejected_even_with_recomputed_manifest(source, tmp_path):
    result = build_release(source, tmp_path / "out")
    prefix = f"personal-ai-trading-copilot-{result.version}/"
    payload = b"private-account-data"
    with zipfile.ZipFile(result.archive) as package:
        contents = {info.filename: package.read(info) for info in package.infolist()}
    manifest = json.loads(contents[prefix + MANIFEST_NAME])
    manifest["files"].append({"path": "data/paper/state.json", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    contents[prefix + MANIFEST_NAME] = json.dumps(manifest).encode()
    contents[prefix + "data/paper/state.json"] = payload
    with zipfile.ZipFile(result.archive, "w") as package:
        for name, content in contents.items():
            package.writestr(name, content)
    with pytest.raises(ReleaseError, match="Unexpected"):
        verify_archive(result.archive)


def test_modified_payload_rejected(source, tmp_path):
    result = build_release(source, tmp_path / "out")
    with zipfile.ZipFile(result.archive) as package:
        contents = {info.filename: package.read(info) for info in package.infolist()}
    name = next(name for name in contents if name.endswith("/README.md"))
    contents[name] = b"tampered file"
    with zipfile.ZipFile(result.archive, "w") as package:
        for name, content in contents.items():
            package.writestr(name, content)
    with pytest.raises(ReleaseError, match="differs"):
        verify_archive(result.archive)


def test_windows_reparse_point_is_a_link():
    class Junction:
        def lstat(self):
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
    assert _is_link(Junction())


def test_linked_source_is_rejected(source, tmp_path, monkeypatch):
    # Simulate the reparse metadata OS traversal reports, without requiring
    # administrator/Developer Mode privileges to create a Windows junction.
    from scripts import build_release as module
    linked = source / "backend" / "linked"
    linked.mkdir()
    write(source, "backend/linked/account.py")
    real = module._is_link
    monkeypatch.setattr(module, "_is_link", lambda path: path == linked or real(path))
    with pytest.raises(ReleaseError, match="Linked source directory"):
        collect_payload(source)


@pytest.mark.parametrize("version", ["../1.1", "1.1/escape", "not-a-version", "1.1\nsecret"])
def test_invalid_version_cannot_control_output_path(source, tmp_path, version):
    write(source, "VERSION", version)
    with pytest.raises(ReleaseError, match="release version"):
        build_release(source, tmp_path / "out")


def test_actual_workspace_archive_imports_homepage_without_network_or_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = build_release(root, tmp_path / "out")
    smoke_test_archive(result.archive)
