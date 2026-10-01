#!/usr/bin/env python3
"""
Download the habitat-ready Matterport3D scenes needed for Stage 0.

WHY THIS EXISTS instead of using the official download_mp.py directly:

1. The official script is PYTHON 2 ONLY. It uses `print` statements,
   `raw_input`, and `urllib.urlretrieve` -- it will not even parse under
   Python 3, and no Python 2 is present in this project's environments.

2. It has a re-run bug in `download_task_data`: the `download_file` call sits
   INSIDE `if not os.path.isdir(localdir):`, so once the output directory
   exists the function silently downloads nothing and prints nothing. Any
   interrupted download would appear to "succeed" on retry while doing nothing.
   A 15 GB download over a home connection gets interrupted.

3. `--id` does NOT apply to task data. `download_task_data` ignores it, so the
   habitat bundle cannot be filtered to specific scans at download time. The
   17-scan filter has to happen at EXTRACTION time, which this script does.

WHICH DATA: `mp3d_habitat.zip` (TASK_FILES['habitat']), not the raw
`matterport_mesh` filetype. The habitat bundle contains the .glb meshes and
.navmesh navigation meshes that habitat-sim actually loads. The raw
matterport_mesh is a different format and is NOT habitat-ready -- downloading
it would mean running the mesh conversion pipeline ourselves.

The whole bundle is one archive covering all 90 scenes (~15 GB, vs 1.3 TB for
the full MP3D release). Only the scans this project needs are EXTRACTED.

USAGE
    # 0. sign the Matterport3D terms of use, then export the habitat-archive link you are sent
    export MP3D_HABITAT_URL=<link from the Matterport3D team>

    # 1. see what will happen, download nothing
    python scripts/download_mp3d.py --dry-run

    # 2. do it (prompts for MP3D terms-of-use acknowledgement)
    python scripts/download_mp3d.py

    # 3. if the download is interrupted, just run it again -- it resumes

    # or fetch only some scans (their .glb + .navmesh, ~80 MB each) without the 15 GB archive
    python scripts/download_mp3d.py --scans 2azQ1b91cZZ zsNo4HB9uLZ
"""

# --- ROS guard: strip /opt/ros/* from sys.path before any other import. ------
import sys
sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
# ----------------------------------------------------------------------------

import argparse
import time
import io
import gzip
import json
import os
import pathlib
import shutil
import urllib.request
import zipfile

# Not published on purpose: the Matterport3D team emails the download link after you sign the terms of use.
# Set MP3D_HABITAT_URL to the habitat task-archive link from that email.
HABITAT_ZIP_URL = os.environ.get("MP3D_HABITAT_URL", "")
TOS_URL = "http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf"

_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = _ROOT / "data" / "scene_datasets"
R2R_DIR = _ROOT / "data" / "R2R_VLNCE_v1-3"

# Files habitat needs per scene. .house carries semantic annotations and is
# optional for RGB navigation, but is bundled anyway.
REQUIRED_SUFFIXES = (".glb", ".navmesh")


# ---------------------------------------------------------------------------
# Which scans do we actually need?
# ---------------------------------------------------------------------------
def required_scans(splits=("train", "val_unseen"), train_episode_budget=1500):
    """Scan ids needed for Stage 0, derived from the R2R-CE episodes.

    Derived rather than hardcoded so it cannot drift from the data. Takes ALL
    val_unseen scans (the gate split must be complete) plus the fewest train
    scans covering `train_episode_budget` episodes -- AGENTS.md Sec. 3.2 asks
    for 1000-2000 training episodes, not the full set.
    """
    from collections import Counter

    needed, detail = set(), {}

    if "val_unseen" in splits:
        eps = _load_split("val_unseen")
        vu = sorted({e["scene_id"].split("/")[1] for e in eps})
        needed |= set(vu)
        detail["val_unseen"] = {"scans": vu, "episodes": len(eps)}

    if "train" in splits:
        eps = _load_split("train")
        counts = Counter(e["scene_id"].split("/")[1] for e in eps)
        chosen, total = [], 0
        for scan, n in counts.most_common():
            if total >= train_episode_budget:
                break
            chosen.append(scan)
            total += n
        needed |= set(chosen)
        detail["train"] = {"scans": sorted(chosen), "episodes": total}

    return sorted(needed), detail


def _load_split(split):
    path = R2R_DIR / split / f"{split}.json.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download R2R_VLNCE_v1-3 into data/ first."
        )
    with gzip.open(path, "rt") as fh:
        return json.load(fh)["episodes"]


# ---------------------------------------------------------------------------
# Resumable download
# ---------------------------------------------------------------------------
def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


class IncompleteDownload(RuntimeError):
    """Transfer ended before Content-Length bytes arrived."""


def remote_size(url):
    """Content-Length for `url`, or None if the server will not say."""
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=60
        ) as r:
            return int(r.headers.get("Content-Length", 0)) or None
    except Exception:
        return None


def download_resumable(url, dest: pathlib.Path, chunk=1 << 20, attempts=5):
    """Download `url` to `dest`, resuming via HTTP Range, verifying the size.

    A dropped connection makes `resp.read()` return b"" WITHOUT raising, so a
    naive loop treats a truncated transfer as success. This function therefore
    never promotes `.part` to `dest` until the byte count matches
    Content-Length, and retries from wherever it stopped.

    An existing `dest` of the wrong size is demoted back to `.part` and
    resumed, rather than being trusted or thrown away.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    total = remote_size(url)

    if dest.exists():
        size = dest.stat().st_size
        if total and size == total:
            print(f"  already complete: {dest} ({_human(total)})")
            return dest
        if total:
            # Truncated from an earlier run -- keep the bytes, resume them.
            print(f"  existing file is {_human(size)}, expected {_human(total)}"
                  f" -- treating as partial and resuming")
            dest.rename(part)

    if total is None:
        print("  WARNING: server gave no Content-Length; cannot verify "
              "completeness. Re-run and confirm the size is stable.")

    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:
            break

        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
            print(f"  [attempt {attempt}/{attempts}] resuming at {_human(have)}")
        else:
            print(f"  [attempt {attempt}/{attempts}] starting")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if have and resp.status != 206:
                    # Range ignored: appending would corrupt the file.
                    print("  server ignored Range; restarting from zero")
                    if part.exists():
                        part.unlink()
                    have = 0
                with open(part, "ab" if have else "wb") as out:
                    done = have
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        out.write(buf)
                        done += len(buf)
                        if total:
                            pct = 100.0 * done / total
                            print(f"\r  {_human(done)} / {_human(total)} "
                                  f"({pct:5.1f}%)", end="", flush=True)
                        else:
                            print(f"\r  {_human(done)}", end="", flush=True)
            print()
        except Exception as exc:                      # noqa: BLE001
            print(f"\n  transfer error: {type(exc).__name__}: {exc}")

        got = part.stat().st_size if part.exists() else 0
        if total and got >= total:
            break
        if total:
            print(f"  incomplete: {_human(got)} of {_human(total)}; retrying")

    got = part.stat().st_size if part.exists() else 0
    if total and got != total:
        raise IncompleteDownload(
            f"got {got} bytes, expected {total}. The partial file is kept at "
            f"{part} -- re-run this script to continue from there."
        )

    part.rename(dest)
    print(f"  complete: {dest} ({_human(got)})")
    return dest


def verify_archive(archive: pathlib.Path, deep: bool = False):
    """Confirm the archive is a readable zip before trying to extract.

    A truncated download still carries a valid PK header -- `file` happily
    reports "Zip archive data" -- so only reading the central directory, which
    lives at the END of the file, actually proves the transfer finished.

    `deep=True` additionally CRC-checks every member. That reads the whole
    archive (~15 GB here, stored uncompressed), so it is off by default: the
    byte-exact Content-Length match plus an intact central directory is already
    strong evidence.
    """
    try:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()          # requires an intact central directory
            if deep:
                bad = zf.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"corrupt member: {bad}")
            return len(names)
    except zipfile.BadZipFile as exc:
        raise IncompleteDownload(
            f"{archive} is not a readable zip ({exc}). It is most likely "
            f"truncated -- delete it and re-run to download again."
        ) from exc


# ---------------------------------------------------------------------------
# Selective extraction
# ---------------------------------------------------------------------------
def extract_scans(archive: pathlib.Path, out_dir: pathlib.Path, scans):
    """Extract ONLY the requested scans from the bundle.

    The archive covers all 90 scenes; pulling out 17 keeps ~4/5 of the disk
    cost off the machine.
    """
    wanted = set(scans)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        members, found = [], set()
        for name in names:
            parts = pathlib.PurePosixPath(name).parts
            hit = next((p for p in parts if p in wanted), None)
            if hit:
                members.append(name)
                found.add(hit)

        missing = wanted - found
        if missing:
            print(f"  WARNING: not present in archive: {sorted(missing)}")

        print(f"  extracting {len(members)} files for {len(found)} scans ...")
        for i, name in enumerate(members, 1):
            zf.extract(name, out_dir)
            if i % 25 == 0 or i == len(members):
                print(f"\r  {i}/{len(members)} files", end="", flush=True)
        print()
    return found


def verify(out_dir: pathlib.Path, scans):
    """Confirm each scan has the files habitat needs, and report .glb paths."""
    print(f"\n{'scan':16s} {'.glb':>6} {'.navmesh':>9}  path")
    print("-" * 78)
    ok = []
    for scan in sorted(scans):
        hits = list(out_dir.rglob(f"{scan}/{scan}.glb"))
        navs = list(out_dir.rglob(f"{scan}/{scan}.navmesh"))
        good = bool(hits) and bool(navs)
        ok.append(good)
        p = hits[0].relative_to(out_dir) if hits else "-"
        print(f"{scan:16s} {'yes' if hits else 'NO':>6} {'yes' if navs else 'NO':>9}  {p}")
    print("-" * 78)
    print(f"{sum(ok)}/{len(scans)} scans complete")
    return all(ok)


# ---------------------------------------------------------------------------
class _HttpRange(io.RawIOBase):
    """Seekable read-only view of a remote file over HTTP Range requests, so zipfile can read single members."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                time.sleep(5 * (attempt + 1))
        else:
            raise IncompleteDownload(f"range read at byte {self.pos} failed 6 times")
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_scans_remote(url, out_dir: pathlib.Path, scans, workers=4):
    """Fetch only the .glb and .navmesh of `scans` from the remote archive, without downloading all 15 GB.

    The archive is a plain zip, and the server honours Range requests (download_resumable relies on the same), so each
    member can be read on its own. Habitat needs only the mesh and the navmesh; the semantic .ply and .house files are
    skipped. Scans are fetched in parallel, and files already complete are skipped, so re-running resumes.
    """
    import concurrent.futures as cf

    def one(scan):
        zf = zipfile.ZipFile(io.BufferedReader(_HttpRange(url), buffer_size=16 << 20))
        got = 0
        for info in zf.infolist():
            name = info.filename
            if f"/{scan}/" not in name or not name.endswith((".glb", ".navmesh")):
                continue
            dest = out_dir / name[name.index("mp3d/"):] if "mp3d/" in name else out_dir / name
            if dest.exists() and dest.stat().st_size == info.file_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with zf.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            tmp.replace(dest)
            got += info.file_size
        return scan, got

    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        for scan, got in ex.map(one, scans):
            print(f"  {scan}: {_human(got)}  ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Download habitat-ready MP3D scenes")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                    help=f"output directory (default {DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be downloaded and exit")
    ap.add_argument("--all-scans", action="store_true",
                    help="extract all 90 scenes instead of only those needed")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep mp3d_habitat.zip after extracting (default: keep)")
    ap.add_argument("--episodes", type=int, default=1500,
                    help="train-episode budget used to pick train scans")
    ap.add_argument("--scans", nargs="+", default=None,
                    help="fetch ONLY these scans' .glb/.navmesh via HTTP Range requests (no 15 GB archive)")
    ap.add_argument("--workers", type=int, default=4, help="parallel connections for --scans")
    ap.add_argument("--accept-terms", action="store_true",
                    help="confirm you have agreed to the Matterport3D terms of use (skips the ENTER prompt)")
    args = ap.parse_args()

    scans, detail = required_scans(train_episode_budget=args.episodes)
    if args.scans:
        scans = sorted(set(args.scans))

    print("=" * 78)
    print("  MP3D habitat-ready scene download")
    print("=" * 78)
    for split, d in detail.items():
        print(f"  {split:11s}: {len(d['scans']):2d} scans, {d['episodes']} episodes")
    print(f"  TOTAL      : {len(scans)} scans (of 90 in the release)")
    print()
    print(f"  archive : {'MP3D_HABITAT_URL (set)' if HABITAT_ZIP_URL else 'MP3D_HABITAT_URL NOT SET'}")
    if args.scans:
        print(f"  approx  : ~{80 * len(scans)} MB (.glb + .navmesh only, fetched from inside the archive)")
    else:
        print(f"  approx  : ~15 GB download (the full MP3D release is 1.3 TB)")
    print(f"  out dir : {args.out}")
    print()
    print("  Scans:")
    for i in range(0, len(scans), 6):
        print("    " + " ".join(scans[i:i + 6]))
    print()

    if args.dry_run:
        print("  --dry-run: nothing downloaded.")
        return 0

    if not HABITAT_ZIP_URL:
        print("  MP3D_HABITAT_URL is not set. Sign the Matterport3D terms of use; the team emails you the\n"
              f"  download link. Terms: {TOS_URL}")
        return 1

    print("*" * 78)
    print("  By continuing you confirm you have agreed to the Matterport3D")
    print("  terms of use:")
    print(f"    {TOS_URL}")
    print("*" * 78)
    if args.accept_terms:
        print("  --accept-terms given: you confirm you have agreed to these terms.")
    else:
        try:
            input("  Press ENTER to continue, or CTRL-C to abort: ")
        except (KeyboardInterrupt, EOFError):
            print("\n  aborted (no terminal input; pass --accept-terms to confirm non-interactively).")
            return 1

    if args.scans:
        print(f"\nFetching {len(scans)} scans (.glb + .navmesh only) with {args.workers} connections:")
        fetch_scans_remote(HABITAT_ZIP_URL, args.out, scans, args.workers)
        return 0 if verify(args.out, scans) else 1

    archive = args.out / "mp3d_habitat.zip"
    print(f"\nDownloading (resumable -- rerun this script if interrupted):")
    try:
        download_resumable(HABITAT_ZIP_URL, archive)
    except IncompleteDownload as exc:
        print(f"\nDOWNLOAD INCOMPLETE: {exc}")
        return 1

    print("\nVerifying archive integrity ...")
    try:
        n_members = verify_archive(archive)
        print(f"  OK: {n_members} entries")
    except IncompleteDownload as exc:
        print(f"  FAILED: {exc}")
        return 1

    print(f"\nExtracting into {args.out} ...")
    if args.all_scans:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(args.out)
        found = scans
    else:
        found = extract_scans(archive, args.out, scans)

    all_ok = verify(args.out, scans)

    print("\nNext step:")
    print("  conda activate habitat_render")
    example = next(iter(sorted(found)), scans[0])
    hits = list(args.out.rglob(f"{example}/{example}.glb"))
    if hits:
        print(f"  python scripts/verify_habitat.py --scene {hits[0]}")
    if not all_ok:
        print("\n  WARNING: some scans are incomplete -- see the table above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
