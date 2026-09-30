"""Export scalar histories for the explicit allowlist; never download artifacts."""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import wandb

ROOT = Path(__file__).resolve().parent

def scalar(value):
    return isinstance(value, (int, float, bool)) and (not isinstance(value, float) or math.isfinite(value))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'raw/wandb/runs')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'run_manifest.json').read_text())
    api = wandb.Api(timeout=120)  # WANDB_API_KEY or an existing wandb login
    names = [r['name'] for r in manifest['runs']]
    found = list(api.runs(manifest['project'], filters={'displayName': {'$in': names}}))
    by_name = {}
    for run in found:
        by_name.setdefault(run.name, []).append(run)
    selected = []
    for entry in manifest['runs']:
        if 'id' in entry:
            run = api.run(f"{manifest['project']}/{entry['id']}")
        else:
            matches = by_name.get(entry['name'], [])
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {entry['name']}, found {len(matches)}; pin its ID in the manifest.")
            run = matches[0]
        if run.name != entry['name']:
            raise ValueError(f"Run name mismatch for {run.id}")
        selected.append((entry, run))
    args.output.mkdir(parents=True, exist_ok=True)
    for entry, run in selected:
        history = []
        for row in run.scan_history(page_size=1000):  # full history, not sampled run.history()
            history.append({k: v for k, v in row.items() if scalar(v)})
        # Preserve all numeric metrics; omit text, media, credentials, paths and artifacts.
        config = {k: v for k, v in run.config.items() if scalar(v) or k in {
            'model_type', 'lr_scheduler_type', 'model_size', 'model', 'report_to'}}
        summary = {k: v for k, v in dict(run.summary).items() if scalar(v)}
        record = {'id': run.id, 'name': run.name, 'group': entry['group'],
                  'url': run.url, 'state': run.state, 'config': config,
                  'summary': summary, 'history': history,
                  'exported_at_utc': datetime.now(timezone.utc).isoformat(),
                  'history_export': 'scan_history, all rows, numeric scalar fields only'}
        (args.output / f'{run.id}.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        print(f'{run.name}: {len(history)} history rows')
    print(f'Exported {len(selected)} allowlisted runs; no artifacts or checkpoints.')

if __name__ == '__main__':
    main()
