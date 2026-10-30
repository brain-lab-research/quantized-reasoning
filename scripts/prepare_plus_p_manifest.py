"""Prepare full Llama W2 +P tasks for one shared-queue worker."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vv.distill_llama_matrix import prepare_tasks
from vv.run_distill_plus_p import W2_ORDER


def build(root: Path):
    tasks = prepare_tasks(root, precision='w2')
    if len(tasks) != 9 or {t['benchmark'] for t in tasks} != set(W2_ORDER):
        raise ValueError('Expected all nine benchmarks')
    tasks.sort(key=lambda t: W2_ORDER.index(t['benchmark']))
    manifest = []
    for task in tasks:
        path = Path(task['config'])
        manifest.append({'config': str(path), 'output': str(path.parent / 'artifacts'),
                         'start': 0, 'count': task['count'], 'benchmark': task['benchmark']})
    path = root / 'plus-p-manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing manifest differs')
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.root)
    print('MANIFEST_READY', len(manifest), sum(t['count'] for t in manifest), flush=True)
