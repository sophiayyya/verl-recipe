import glob
import os
import sys
from pathlib import Path

# A line ending in "\" splices the NEXT line onto it. If that next line is a
# comment, bash swallows the rest of the command: every argument below is
# silently dropped. Same class of bug found in train_qwen3_30b_sglang.sh.
RECIPE_ROOT = Path(__file__).resolve().parents[1]
pats = sys.argv[1:] or [str(RECIPE_ROOT / "**" / "*.sh"), str(RECIPE_ROOT / "**" / "*.sbatch")]
files = []
for p in pats:
    files.extend(glob.glob(p, recursive=True))

bad = 0
for f in sorted(set(files)):
    if not os.path.isfile(f):
        continue
    try:
        lines = open(f, errors="replace").read().split("\n")
    except Exception:
        continue
    for i in range(len(lines) - 1):
        cur = lines[i].rstrip()
        if not cur.endswith("\\"):
            continue
        if cur.endswith("\\\\"):
            continue
        nxt = lines[i + 1].lstrip()
        if nxt == "":
            bad += 1
            print(f"{f}:{i + 2}: BLANK line inside continuation chain (command ends here; args below are dropped)")
            print(f"    {i + 1}| {cur[:110]}")
            continue
        if nxt.startswith("#"):
            bad += 1
            print(f"{f}:{i + 2}: continuation swallowed by comment")
            print(f"    {i + 1}| {cur[:110]}")
            print(f"    {i + 2}| {lines[i + 1][:110]}")

print(f"TOTAL_BROKEN={bad}  files_scanned={len(set(files))}")
