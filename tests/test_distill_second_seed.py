import yaml
from scripts.prepare_distill_second_seed import prepare


def test_second_seed_preserves_dataset_and_has_independent_outputs(tmp_path):
    tasks=prepare(tmp_path)
    assert len(tasks)==9 and sum(t['count'] for t in tasks)==9387
    for t in tasks:
        cfg=yaml.safe_load(open(t['config']))
        assert cfg['benchmark']['seed']==42
        assert cfg['actors'][0]['generation_params']['seed']==43
        assert 'seed43' in cfg['experiment_name']
    assert prepare(tmp_path)==tasks
