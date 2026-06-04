import json, pathlib, sys

results_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")

found = False
for f in sorted(results_dir.glob("F[BDQ]_*.txt")):
    lines = f.read_text().strip().split("\n")
    for i, line in enumerate(lines):
        if line.startswith("======== ROUND"):
            try:
                scores = json.loads(lines[i+3])
                if all(s == 0.0 for s in scores):
                    print(f"ALL-ZERO: {f.name} {line} -> {scores}")
                    found = True
                elif any(s == 0.0 for s in scores):
                    print(f"HAS-ZERO: {f.name} {line} -> {scores}")
                    found = True
            except Exception:
                pass

if not found:
    print("All clean. No zero scores found.")
