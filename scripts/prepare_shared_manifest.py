"""Create a full ordered native-chat W2 manifest for one shared-queue owner."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vv.distill_llama_matrix import prepare_tasks

ORDER = ['gpqa_diamond', 'aime2026', 'math500', 'strategyqa', 'arc_challenge',
         'gsm8k', 'piqa', 'arc_easy', 'winogrande']


def build(root):
    tasks = prepare_tasks(root, precision='w2')
    if len(tasks) != len(ORDER) or {t['benchmark'] for t in tasks} != set(ORDER):
        raise ValueError('Expected all nine full benchmarks')
    tasks.sort(key=lambda t: ORDER.index(t['benchmark']))
    manifest = [{'config': t['config'], 'output': str(Path(t['config']).parent / 'artifacts'),
                 'start': 0, 'count': t['count']} for t in tasks]
    path = root / 'shared-manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing shared manifest differs')
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    manifest = build(a.root)
    print('MANIFEST_READY', len(manifest), sum(t['count'] for t in manifest), flush=True)


if __name__ == '__main__':
    main()
