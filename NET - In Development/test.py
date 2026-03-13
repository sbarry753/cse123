import json

manifest_path = "./dataset/manifest.jsonl"
manifest = [json.loads(l) for l in open(manifest_path)]

for entry in manifest:
    lab = json.load(open(f"./dataset/{entry['label']}"))
    lab_source = lab.get('source', '')
    if 'idmt' in lab_source:
        entry['source'] = 'idmt'
    elif 'guitarset' in lab_source.lower():
        entry['source'] = 'guitarset'
    else:
        entry['source'] = 'di'

# Write back
with open(manifest_path, 'w') as f:
    for entry in manifest:
        f.write(json.dumps(entry) + '\n')

# Verify
from collections import Counter
print(Counter(e['source'] for e in manifest))