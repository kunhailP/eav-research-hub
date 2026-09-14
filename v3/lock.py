"""Record a preregistration lock: sha256 of the listed files plus UTC time in docs/decisions/LOCKS.md.

  python v3/lock.py --name prereg-v3-uk docs/13_prereg_v3_draft.md label/tasks/v3/*.json
  python v3/lock.py --check prereg-v3-uk     # exit 1 if any locked file changed or the lock is missing
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "docs/decisions/LOCKS.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_locks():
    if not LOCKS.exists():
        return {}
    locks = {}
    for line in LOCKS.read_text().splitlines():
        if line.startswith("<!-- lock "):
            entry = json.loads(line[len("<!-- lock "):-len(" -->")])
            locks[entry["name"]] = entry
    return locks


def is_locked(name: str) -> bool:
    entry = read_locks().get(name)
    return bool(entry) and all((ROOT / f).exists() and digest(ROOT / f) == h for f, h in entry["files"].items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="lock name to create")
    ap.add_argument("--check", help="lock name to verify")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    if a.check:
        ok = is_locked(a.check)
        print(f"{a.check}: {'intact' if ok else 'MISSING OR CHANGED'}")
        return 0 if ok else 1
    if not a.name or not a.files:
        ap.error("--name and files are required to create a lock")
    if a.name in read_locks():
        ap.error(f"lock {a.name!r} already exists; locks are append-only, use a new name")
    files = {str(Path(f).resolve().relative_to(ROOT)): digest(Path(f)) for f in a.files}
    entry = {"name": a.name, "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "files": files}
    header = "" if LOCKS.exists() else "# Preregistration locks (append-only)\n\n"
    with LOCKS.open("a") as fh:
        fh.write(header + f"## {a.name} · {entry['utc']}\n\n" + "\n".join(f"- `{f}` sha256 `{h}`" for f, h in files.items())
                 + f"\n\n<!-- lock {json.dumps(entry)} -->\n\n")
    print(f"locked {a.name}: {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
