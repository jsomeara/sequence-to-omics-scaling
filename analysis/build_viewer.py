"""Build a standalone offline HTML viewer from the allowlisted scalar run exports."""
import base64
import gzip
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT/'run_manifest.json').read_text())
runs = []
for entry in manifest['runs']:
    run = json.loads((ROOT/'raw/wandb/runs'/f"{entry['id']}.json").read_text())
    assert run['name'] == entry['name'] and run['group'] == entry['group']
    runs.append(run)
payload = base64.b64encode(gzip.compress(json.dumps(runs, separators=(',', ':'), allow_nan=False).encode(), mtime=0)).decode()
template = (ROOT/'viewer/template.html').read_text()
(ROOT.parent/'run-viewer.html').write_text(template.replace('__RUN_DATA__',payload))
print(f'Built run-viewer.html: {len(runs)} runs, {sum(len(r["history"]) for r in runs):,} rows')
