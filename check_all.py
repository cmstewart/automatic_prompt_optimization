import json, pathlib, sys

results_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("results/experiments")

suspect = []
clean = 0
total = 0

for f in sorted(results_dir.glob("F[BDQ]_*.txt")):
    lines = f.read_text().strip().split("\n")
    all_zero_rounds = 0
    total_rounds = 0
    for i, line in enumerate(lines):
        if line.startswith("======== ROUND"):
            try:
                scores = json.loads(lines[i+3])
                total_rounds += 1
                if all(s == 0.0 for s in scores):
                    all_zero_rounds += 1
            except Exception:
                pass
    total += 1
    if total_rounds < 7:
        suspect.append(f"{f.name}: INCOMPLETE ({total_rounds}/7 rounds)")
    elif all_zero_rounds >= 3:
        suspect.append(f"{f.name}: LIKELY CORRUPTED ({all_zero_rounds}/7 all-zero rounds)")
    elif all_zero_rounds >= 1:
        suspect.append(f"{f.name}: POSSIBLY OK ({all_zero_rounds}/7 all-zero rounds, could be small test set)")
    else:
        clean += 1

print(f"Total files: {total}")
print(f"Clean: {clean}")
print(f"Flagged: {len(suspect)}")
print()
for s in suspect:
    print(f"  {s}")
