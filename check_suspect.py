import json, pathlib

for f in sorted(pathlib.Path('results/experiments').glob('FB_*.txt')):
    lines = f.read_text().strip().split('\n')
    all_zero_count = 0
    total_rounds = 0
    for i, line in enumerate(lines):
        if line.startswith('======== ROUND'):
            try:
                scores = json.loads(lines[i+3])
                total_rounds += 1
                if all(s == 0.0 for s in scores):
                    all_zero_count += 1
            except:
                pass
    if all_zero_count >= 3:
        print(f'SUSPECT: {f.name} ({all_zero_count}/{total_rounds} all-zero rounds)')

print('Done.')
