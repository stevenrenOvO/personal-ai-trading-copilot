"""Build a deterministic source release from saved workspace files, without Git.

Only the explicit source allowlist below is packaged. Runtime data, credentials,
links and machine state are never traversed. No network or Git command is used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import zipfile


PROJECT_NAME = "personal-ai-trading-copilot"
MANIFEST_NAME = "RELEASE-MANIFEST.json"
ROOT_FILES = frozenset({
    "VERSION", "LICENSE", "README.md", "requirements.txt", ".env.example", ".gitignore",
    "requirements-runtime.txt", "requirements-dev.txt", "requirements.lock.txt",
    "start.cmd", "start.ps1", "stop.cmd", "stop.ps1", "setup.cmd", "setup.ps1", "start-demo.cmd", "stop-demo.cmd",
})
SOURCE_DIRS = frozenset({"backend", "scripts", "tests", "features", "backtest", "journal", "risk", "strategies", "docs"})
FIXTURES = frozenset({f"tests/fixtures/{name}.csv" for name in ("daily", "daily_basic", "stock_basic", "stk_limit")})
STATIC_SUFFIXES = frozenset({".html", ".css", ".js", ".svg", ".png", ".jpg", ".jpeg", ".ico", ".woff", ".woff2"})
EXCLUDED_DIRS = frozenset({
    ".git", ".venv", "venv", "env", ".agents", ".codex", ".aws", ".ssh", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".tox", ".nox",
    "tmp", "temp", "logs", "cache", "dist", "build", "_build", "secrets", "credentials",
})
SECRET_NAME = re.compile(r"(?:^|[._-])(?:secret|secrets|credential|credentials)(?:[._-]|$)", re.I)
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:-[0-9A-Za-z.-]+)?")
WINDOWS_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", re.I)
REQUIRED_FILES = frozenset({
    "VERSION", "LICENSE", "README.md", "requirements.txt", "backend/api/app.py",
    "backend/dashboard/static/index.html", "start.cmd", "start.ps1", "stop.cmd", "stop.ps1",
})
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_RELEASE_BYTES = 256 * 1024 * 1024


class ReleaseError(ValueError):
    """A release contains an unsafe path, unexpected file, or invalid digest."""


@dataclass(frozen=True)
class ReleaseResult:
    version: str
    archive: Path
    checksum_file: Path
    sha256: str
    file_count: int


def _safe_relative_path(name: str) -> PurePosixPath:
    # Raw checks matter: PurePosixPath normalizes away '.' and repeated '/'.
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ReleaseError("Invalid release path")
    parts = name.split("/")
    if any(part in ("", ".", "..") or ":" in part or part.endswith((" ", "."))
           or any(ord(c) < 32 for c in part) or WINDOWS_RESERVED.fullmatch(part) for part in parts):
        raise ReleaseError(f"Unsafe release path: {name!r}")
    result = PurePosixPath(name)
    if result.is_absolute():
        raise ReleaseError(f"Absolute release path: {name!r}")
    return result


def allowed_payload(name: str) -> bool:
    path = _safe_relative_path(name)
    if name in ROOT_FILES:
        return True
    if path.parts[0] not in SOURCE_DIRS or len(path.parts) < 2:
        return False
    lowered = [part.lower() for part in path.parts]
    if any(part in EXCLUDED_DIRS or part.startswith(".env") or SECRET_NAME.search(part) for part in lowered):
        return False
    if name in FIXTURES:
        return True
    if path.parts[:2] == ("tests", "fixtures"):
        return False
    suffix = path.suffix.lower()
    if path.parts[0] == "docs":
        return suffix in {".md", ".py"} or name == "docs/V1.1-research-report.json"
    if path.parts[:3] == ("backend", "dashboard", "static"):
        return suffix in STATIC_SUFFIXES
    if path.parts[0] == "scripts":
        return suffix in {".py", ".ps1", ".cmd"}
    if path.parts[0] == "tests":
        return suffix in {".py", ".cjs"}
    return suffix == ".py"


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def collect_payload(root: Path) -> dict[str, bytes]:
    root = Path(root).resolve(strict=True)
    payload: dict[str, bytes] = {}
    seen: set[str] = set()

    def add(path: Path) -> None:
        name = path.relative_to(root).as_posix()
        if not allowed_payload(name):
            return
        if _is_link(path) or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ReleaseError(f"Source must be a regular file inside the workspace: {name}")
        if name.casefold() in seen:
            raise ReleaseError(f"Case-insensitive filename collision: {name}")
        if path.stat().st_size > MAX_MEMBER_BYTES:
            raise ReleaseError(f"Source file exceeds release size limit: {name}")
        payload[name] = path.read_bytes()
        seen.add(name.casefold())

    for name in sorted(ROOT_FILES):
        path = root / name
        if path.exists() or path.is_symlink():
            add(path)
    for directory in sorted(SOURCE_DIRS):
        base = root / directory
        if not base.exists() and not base.is_symlink():
            continue
        if _is_link(base):
            raise ReleaseError(f"Linked source directory is not allowed: {directory}")
        for current, dirs, files in os.walk(base, followlinks=False):
            # Prune before recursion: data outside the source allowlist is never visited.
            retained = []
            for child in sorted(dirs):
                if child.lower() in EXCLUDED_DIRS or child.lower().startswith(".env") or SECRET_NAME.search(child):
                    continue
                path = Path(current) / child
                if _is_link(path):
                    raise ReleaseError(f"Linked source directory is not allowed: {path.relative_to(root).as_posix()}")
                retained.append(child)
            dirs[:] = retained
            for child in sorted(files):
                add(Path(current) / child)
    missing = REQUIRED_FILES - payload.keys()
    if missing:
        raise ReleaseError(f"Required files are missing: {', '.join(sorted(missing))}")
    if sum(map(len, payload.values())) > MAX_RELEASE_BYTES:
        raise ReleaseError("Release exceeds total size limit")
    return dict(sorted(payload.items()))


def _version(raw: bytes) -> str:
    try:
        version = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ReleaseError("VERSION must be UTF-8") from exc
    if not VERSION_PATTERN.fullmatch(version):
        raise ReleaseError("VERSION must contain a safe release version, for example 1.1")
    return version


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_archive(archive: Path) -> dict:
    """Verify all paths, the exact allowlist and every payload size/hash before extraction.

    The manifest lists payload files only; the external ZIP SHA256 covers the
    manifest itself. Digests detect changes and are not a publisher signature.
    """
    with zipfile.ZipFile(archive) as package:
        members: dict[str, zipfile.ZipInfo] = {}
        total = 0
        for info in package.infolist():
            _safe_relative_path(info.filename)
            if info.filename.casefold() in {name.casefold() for name in members}:
                raise ReleaseError("Duplicate archive member")
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ReleaseError("Archive members must be regular files")
            total += info.file_size
            if info.file_size > MAX_MEMBER_BYTES or total > MAX_RELEASE_BYTES:
                raise ReleaseError("Archive exceeds release size limit")
            members[info.filename] = info
        manifests = [name for name in members if name.endswith("/" + MANIFEST_NAME)]
        if len(manifests) != 1:
            raise ReleaseError("Archive must contain exactly one release manifest")
        manifest_path = manifests[0]
        manifest = json.loads(package.read(manifest_path))
        if not isinstance(manifest, dict) or manifest.get("schema") != 1 or manifest.get("hash_algorithm") != "sha256":
            raise ReleaseError("Unsupported release manifest")
        version = _version(str(manifest.get("version", "")).encode("utf-8"))
        prefix = f"{PROJECT_NAME}-{version}/"
        if manifest_path != prefix + MANIFEST_NAME:
            raise ReleaseError("Release root does not match version")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ReleaseError("Manifest must list payload files")
        expected: set[str] = set()
        for record in files:
            if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
                raise ReleaseError("Invalid manifest record")
            name = record["path"]
            if not allowed_payload(name) or name.casefold() in expected:
                raise ReleaseError(f"Unexpected or duplicate payload file: {name}")
            expected.add(name.casefold())
            member = prefix + name
            if member not in members:
                raise ReleaseError(f"Payload file missing: {name}")
            if record["size"] != members[member].file_size or record["sha256"] != _digest(package.read(member)):
                raise ReleaseError(f"Payload digest or size differs: {name}")
        actual = {name.removeprefix(prefix).casefold() for name in members if name != manifest_path and name.startswith(prefix)}
        if len(members) != len(expected) + 1 or actual != expected or not {name.casefold() for name in REQUIRED_FILES}.issubset(expected):
            raise ReleaseError("Archive contents differ from manifest or required files")
        if _version(package.read(prefix + "VERSION")) != version:
            raise ReleaseError("VERSION differs from manifest")
        return manifest


def smoke_test_archive(archive: Path) -> None:
    """Check static routes and explicit offline demo data in an isolated process.

    It uses the current Python's installed dependencies, forbids socket connect,
    and never starts a server, live quote service or background data ingestion.
    """
    manifest = verify_archive(archive)
    with tempfile.TemporaryDirectory(prefix="copilot-release-check-") as directory:
        with zipfile.ZipFile(archive) as package:
            package.extractall(directory)  # Every member was validated above.
        root = Path(directory) / f"{PROJECT_NAME}-{manifest['version']}"
        code = '''import json, os, socket, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
def forbidden(*args, **kwargs):
    raise AssertionError("Release smoke check attempted a network/runtime call")
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
from backend.api import app
original_factory = app.Runtime.__dict__["create"]
app.Runtime.create = forbidden
for path in ("/", "/health"):
    status = []
    body = b"".join(app.application({"PATH_INFO": path, "REQUEST_METHOD": "GET"}, lambda s, h: status.append(s)))
    assert status == ["200 OK"], (path, status)
    if path == "/":
        assert b"<html" in body.lower() and b"</html>" in body.lower()
    else:
        assert json.loads(body)["version"] == open("VERSION", encoding="utf-8").read().strip()
assert app._RUNTIME is None
assert not os.path.exists("data"), "Smoke check created runtime data"
print("Homepage and health import: OK (network/runtime disabled)")
app.Runtime.create = original_factory
os.environ["COPILOT_DEMO_MODE"] = "1"
os.environ["COPILOT_DB_PATH"] = os.path.join(os.getcwd(), "data", "demo", "market.db")
from backend.data.config import clear_settings_cache
clear_settings_cache()
from backend.api.demo import configure_demo_services
configure_demo_services()
app._RUNTIME = app.Runtime.create()
for path in ("/data/latest", "/market/state", "/opportunities/plan", "/ai/copilot", "/ready"):
    status = []
    body = b"".join(app.application({"PATH_INFO": path, "REQUEST_METHOD": "GET"}, lambda s,h: status.append(s)))
    assert status == ["200 OK"], (path, status, body[:500])
    data = json.loads(body)
    if path == "/market/state":
        assert data["data"]["total_count"] > 0 and data["meta"]["is_fixture"]
    if path == "/ready":
        assert data["data_mode"] == "demo"
assert app._RUNTIME.root == __import__("pathlib").Path(os.environ["COPILOT_DB_PATH"]).parent
print("Extracted offline demo data routes: OK (network disabled)")
'''
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith(("COPILOT_", "TUSHARE_", "PYTHONPATH", "PYTHONHOME"))}
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=root, env=env,
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise ReleaseError("Extracted homepage/health check failed:\n" + result.stdout + result.stderr)


def build_release(root: Path, output_dir: Path, *, smoke: bool = False) -> ReleaseResult:
    payload = collect_payload(root)
    version = _version(payload["VERSION"])
    prefix = f"{PROJECT_NAME}-{version}"
    manifest = {"schema": 1, "version": version, "hash_algorithm": "sha256",
                "files": [{"path": name, "size": len(content), "sha256": _digest(content)}
                          for name, content in payload.items()]}
    contents = {**payload, MANIFEST_NAME: _json_bytes(manifest)}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{prefix}.zip"
    descriptor, temp_name = tempfile.mkstemp(prefix=".release-", suffix=".zip", dir=output_dir)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
            for name, content in sorted(contents.items()):
                info = zipfile.ZipInfo(f"{prefix}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                package.writestr(info, content, compresslevel=9)
        verify_archive(temporary)
        if smoke:
            smoke_test_archive(temporary)
        checksum = _digest(temporary.read_bytes())
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    checksum_file = archive.with_suffix(".zip.sha256")
    checksum_file.write_text(f"{checksum}  {archive.name}\n", encoding="ascii", newline="\n")
    return ReleaseResult(version, archive, checksum_file, checksum, len(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Saved project source root")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: <root>/dist)")
    parser.add_argument("--verify", type=Path, help="Verify an existing archive instead of building")
    parser.add_argument("--smoke", action="store_true", help="Verify extracted homepage/health with network disabled")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            manifest = verify_archive(args.verify)
            if args.smoke:
                smoke_test_archive(args.verify)
            print(f"Verified {args.verify.name}: {manifest['version']}, {len(manifest['files'])} payload files")
        else:
            result = build_release(args.root, args.output_dir or args.root / "dist", smoke=args.smoke)
            print(f"Archive: {result.archive}\nSHA256: {result.sha256}\nChecksum: {result.checksum_file}\nPayload files: {result.file_count}")
        return 0
    except (ReleaseError, OSError, zipfile.BadZipFile, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Release check failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
