import copy
import json

from scripts.collect_multi_bench_traces import scheduler_compatible, repair_trailing_jsonl


def test_only_scheduler_resources_may_change():
    original={'actors':[{'model_id':'same','generation_params':{'seed':43},'backend_kwargs':{'max_num_seqs':16,'kv_cache_memory_bytes':40,'gpu_memory_utilization':0.33,'max_model_len':36864}}], 'benchmark':{'seed':42}}
    modified=copy.deepcopy(original)
    modified['actors'][0]['backend_kwargs'].update(max_num_seqs=32,kv_cache_memory_bytes=80,gpu_memory_utilization=0.6)
    assert scheduler_compatible(original,modified)
    modified['actors'][0]['generation_params']['seed']=42
    assert not scheduler_compatible(original,modified)


def test_interrupted_jsonl_tail_is_preserved_before_repair(tmp_path):
    path=tmp_path/'traces.jsonl'; path.write_bytes(b'{"id":1}\n{"id":')
    repair_trailing_jsonl(path)
    assert path.read_bytes()==b'{"id":1}\n'
    backups=list(tmp_path.glob('.traces.jsonl.interrupted-*'))
    assert len(backups)==1 and backups[0].read_bytes()==b'{"id":'
    path.write_bytes(b'{"id":1}')
    repair_trailing_jsonl(path)
    assert json.loads(path.read_text())=={'id':1} and path.read_bytes().endswith(b'\n')
