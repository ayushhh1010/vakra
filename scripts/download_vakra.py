"""Download only the parts of VAKRA you need, with a size plan first.

`make download` in the VAKRA repo pulls everything (~39.5 GB). Most of that is a
few huge files: six domain databases are ~19 GB, the capability-4 document index
is ~4.4 GB, and three capability-2 ground-truth files are ~3.1 GB. This script
lists the repo, shows exactly what a selection costs, and downloads into the
same folder layout VAKRA expects (<vakra>/data/train/..., <vakra>/data/databases/...).

Nothing is downloaded unless you pass --yes.

Examples
    # 1. See what the training split costs, skipping single files over 50 MB
    python scripts/download_vakra.py --parts train --max-file-mb 50

    # 2. Download it
    python scripts/download_vakra.py --parts train --max-file-mb 50 --yes

    # 3. Per-domain database sizes, smallest first
    python scripts/download_vakra.py --list-db-sizes

    # 4. Databases for small domains only (needed to run the Docker tool servers)
    python scripts/download_vakra.py --parts databases --db-max-domain-mb 300 --yes
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ID = "ibm-research/VAKRA"
CAP_DIRS = {1: "capability_1_bi_apis", 2: "capability_2_dashboard_apis",
            3: "capability_3_multihop_reasoning", 4: "capability_4_multiturn"}
SKIP_NAMES = {".DS_Store", ".gitattributes"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def list_files(token: str | None) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi, RepoFile
    api = HfApi()
    files = []
    for item in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True, token=token):
        if isinstance(item, RepoFile):
            size = item.size or (item.lfs.size if getattr(item, "lfs", None) else 0)
            files.append((item.path, int(size or 0)))
    return files


def db_sizes(files: list[tuple[str, int]]) -> dict[str, int]:
    sizes: dict[str, int] = defaultdict(int)
    for path, size in files:
        parts = path.split("/")
        if parts[0] == "databases" and len(parts) >= 3:
            sizes[parts[1]] += size
    return dict(sizes)


def select(files, parts, caps, domains, train_sides, max_file_mb, db_max_domain_mb):
    chosen, skipped = [], []
    dbs = db_sizes(files)
    max_file = max_file_mb * 1024 ** 2 if max_file_mb else None
    for path, size in files:
        p = path.split("/")
        name = p[-1]
        if name in SKIP_NAMES:
            continue
        top = p[0]
        if top == "train" and "train" in parts and len(p) >= 4:
            cap = next((c for c, d in CAP_DIRS.items() if d == p[1]), None)
            side, domain = p[2], Path(name).stem
            if cap not in caps or side not in train_sides:
                continue
            if domains and domain not in domains:
                continue
            if max_file and size > max_file:
                skipped.append((path, size, f"file > {max_file_mb} MB"))
                continue
            chosen.append((path, size))
        elif top == "databases" and "databases" in parts:
            if len(p) == 2:  # databases/domains.json and helper scripts: always tiny, always take
                chosen.append((path, size))
                continue
            domain = p[1]
            if domains and domain not in domains:
                continue
            if db_max_domain_mb and dbs.get(domain, 0) > db_max_domain_mb * 1024 ** 2:
                skipped.append((path, size, f"domain database > {db_max_domain_mb} MB"))
                continue
            chosen.append((path, size))
        elif top == "indexed_documents" and "indexed_documents" in parts:
            chosen.append((path, size))  # one ChromaDB store for capability 4: all or nothing
    return chosen, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="../vakra/data", help="VAKRA's data folder (default ../vakra/data)")
    ap.add_argument("--parts", nargs="+", default=["train"],
                    choices=["train", "databases", "indexed_documents"])
    ap.add_argument("--capabilities", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    ap.add_argument("--domains", nargs="+", default=None, help="limit to these domains")
    ap.add_argument("--train-sides", nargs="+", default=["input", "output"], choices=["input", "output"])
    ap.add_argument("--max-file-mb", type=float, default=50, help="skip single train files bigger than this (0 = no limit)")
    ap.add_argument("--db-max-domain-mb", type=float, default=0, help="skip domains whose database is bigger (0 = no limit)")
    ap.add_argument("--list-db-sizes", action="store_true")
    ap.add_argument("--token", default=None, help="HF token (optional; the public split needs none)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--yes", action="store_true", help="actually download (otherwise only print the plan)")
    args = ap.parse_args()

    print(f"Listing {REPO_ID} ...")
    files = list_files(args.token)
    print(f"  {len(files)} files, {human(sum(s for _, s in files))} in total\n")

    if args.list_db_sizes:
        for dom, size in sorted(db_sizes(files).items(), key=lambda kv: kv[1]):
            print(f"  {dom:<32} {human(size):>10}")
        return

    chosen, skipped = select(files, set(args.parts), set(args.capabilities),
                             set(args.domains) if args.domains else None, set(args.train_sides),
                             args.max_file_mb, args.db_max_domain_mb)
    total = sum(s for _, s in chosen)
    by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path, size in chosen:
        key = "/".join(path.split("/")[:2]) if not path.startswith("databases/") else "databases"
        by_group[key][0] += 1
        by_group[key][1] += size
    print("Plan")
    for key, (n, size) in sorted(by_group.items()):
        print(f"  {key:<48} {n:>5} files {human(size):>10}")
    print(f"  {'TOTAL':<48} {len(chosen):>5} files {human(total):>10}")
    if skipped:
        print(f"\nSkipped {len(skipped)} files ({human(sum(s for _, s, _ in skipped))}):")
        for path, size, why in sorted(skipped, key=lambda x: -x[1])[:25]:
            print(f"  {path:<70} {human(size):>10}  ({why})")
        if len(skipped) > 25:
            print(f"  ... and {len(skipped) - 25} more")

    data_dir = Path(args.data_dir).resolve()
    probe = data_dir if data_dir.exists() else next(p for p in data_dir.parents if p.exists())
    free = shutil.disk_usage(probe).free
    print(f"\nTarget: {data_dir}\nFree space on that drive: {human(free)}")
    if total * 1.1 > free:
        print("WARNING: not enough free space for this plan (with a 10% margin). Narrow it down.")
        if args.yes:
            sys.exit(1)
    if not args.yes:
        print("\nDry run only. Add --yes to download.")
        return

    from huggingface_hub import snapshot_download
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=str(data_dir),
                      allow_patterns=[p for p, _ in chosen], max_workers=args.workers,
                      token=args.token)
    print(f"\nDone. Files are under {data_dir}")


if __name__ == "__main__":
    main()
