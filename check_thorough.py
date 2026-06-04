import json, pathlib, sys

results_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("results/experiments")

issues = []
total = 0
expected = set()

# Generate all 135 expected filenames
for prefix, ds in [("FB", "financebench"), ("FD", "findoc"), ("FQ", "finqa")]:
    for ev in ["ucb", "ppo", "dpo"]:
        for b in [1, 2, 3]:
            for s in [1, 2, 3, 4, 5]:
                expected.add(f"{prefix}_{ev}_B{b}_s{s}_{ds}_{ev}.txt")

# Check for missing files
found_files = set(f.name for f in results_dir.glob("F[BDQ]_*.txt"))
missing = expected - found_files
for m in sorted(missing):
    issues.append(f"MISSING: {m}")

# Check each file
for f in sorted(results_dir.glob("F[BDQ]_*.txt")):
    total += 1
    lines = f.read_text().strip().split("\n")

    # Check 1: Has 7 rounds
    round_count = sum(1 for l in lines if l.startswith("======== ROUND"))
    if round_count < 7:
        issues.append(f"INCOMPLETE: {f.name} ({round_count}/7 rounds)")
        continue

    # Check 2: All-zero rounds
    all_zero_rounds = 0
    has_zero_rounds = 0
    baseline_zero = False
    final_zero = False
    round_num = 0
    for i, line in enumerate(lines):
        if line.startswith("======== ROUND"):
            try:
                scores = json.loads(lines[i+3])
                if all(s == 0.0 for s in scores):
                    all_zero_rounds += 1
                    if round_num == 0:
                        baseline_zero = True
                round_num += 1
            except Exception:
                issues.append(f"PARSE ERROR: {f.name} at line {i}")

    # Get final round scores
    for i in range(len(lines)-1, -1, -1):
        if lines[i].startswith("[") and "0." in lines[i]:
            try:
                final_scores = json.loads(lines[i])
                if all(s == 0.0 for s in final_scores):
                    final_zero = True
                break
            except:
                pass

    # Check 3: Baseline 0.0 is suspicious for FinQA/FinDoc (large test sets)
    if baseline_zero and not f.name.startswith("FB_"):
        issues.append(f"ZERO BASELINE (non-FB): {f.name}")

    # Check 4: Many all-zero rounds
    if all_zero_rounds >= 3:
        issues.append(f"LIKELY CORRUPTED: {f.name} ({all_zero_rounds}/7 all-zero rounds)")

    # Check 5: Final round all zeros
    if final_zero:
        issues.append(f"FINAL ALL-ZERO: {f.name}")

    # Check 6: Very short file (might have been truncated)
    if len(lines) < 15:
        issues.append(f"SUSPICIOUSLY SHORT: {f.name} ({len(lines)} lines)")

print(f"Expected files: {len(expected)}")
print(f"Found files: {total}")
print(f"Missing files: {len(missing)}")
print()

if issues:
    print(f"ISSUES FOUND: {len(issues)}")
    for issue in sorted(issues):
        print(f"  {issue}")
else:
    print("ALL CLEAN. No issues found.")
