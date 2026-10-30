"""Prepare full benchmark manifests with independent decoding seed 43."""
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.prepare_shared_manifest import ORDER
from vv.distill_llama_matrix import prepare_tasks


def prepare(root, seed=43):
    source=prepare_tasks(root/'template',precision='w2')
    out=[]
    for task in source:
        raw=yaml.safe_load(Path(task['config']).read_text())
        raw['actors'][0]['generation_params']['seed']=seed
        raw['experiment_name']='1030-distill-'+task['benchmark']+'-w2-seed'+str(seed)
        target=root/raw['experiment_name']/'cfg.yaml'
        target.parent.mkdir(parents=True,exist_ok=True)
        raw['output']['base_dir']=str(target.parent/'artifacts')
        if target.exists() and yaml.safe_load(target.read_text())!=raw:
            raise ValueError(f'Existing seed configuration differs: {target}')
        target.write_text(yaml.safe_dump(raw,sort_keys=False))
        out.append({'config':str(target),'output':str(target.parent/'artifacts'),'start':0,'count':task['count'],'benchmark':task['benchmark']})
    out.sort(key=lambda x: ORDER.index(x['benchmark']))
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=out:raise ValueError('Existing manifest differs')
    path.write_text(json.dumps(out,indent=2))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=43);a=p.parse_args()
    tasks=prepare(a.root,a.seed);print('SEED_MANIFEST_READY',a.seed,len(tasks),sum(t['count'] for t in tasks),flush=True)


if __name__=='__main__':main()
