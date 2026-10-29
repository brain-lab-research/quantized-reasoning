import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(os.name != 'posix', reason='requires POSIX shell')
def test_priority_rewires_before_cancelling_and_queues_w4_after_audits(tmp_path):
    base=tmp_path/'data/projects/low-bit-reasoning/exps'
    shared=base/'1029-distill-shared-w2'; w4=base/'1029-w4-32b-full'
    shared.mkdir(parents=True); w4.mkdir(parents=True)
    for name,content in {'gate-job.txt':'23264\n','jobs-a.txt':'23265\n23266\n23267\n',
                         'jobs-b.txt':'23268\n23269\n23270\n','audit-jobs.txt':'23271\n23272\n'}.items():
        (shared/name).write_text(content)
    (w4/'jobs-a-after-distill.txt').write_text('22742\n22743\n22744\n')
    (w4/'jobs-b-after-distill.txt').write_text('22745\n22746\n22747\n')
    bindir=tmp_path/'bin'; bindir.mkdir(); log=tmp_path/'calls.log'; counter=tmp_path/'counter'
    counter.write_text('1000')
    mocks={
      'sbatch':'n=$(cat "$MOCK_COUNTER"); n=$((n+1)); echo "$n" > "$MOCK_COUNTER"; echo "sbatch $*" >> "$MOCK_LOG"; echo "$n"',
      'squeue':'echo "squeue $*" >> "$MOCK_LOG"; [[ "$1" != -h ]] || echo PENDING',
      'scontrol':'echo "scontrol $*" >> "$MOCK_LOG"',
      'scancel':'echo "scancel $*" >> "$MOCK_LOG"',
    }
    for name,body in mocks.items():
        path=bindir/name; path.write_text('#!/bin/bash\nset -e\n'+body+'\n'); path.chmod(0o755)
    env=os.environ.copy(); env.update(PATH=str(bindir)+os.pathsep+env['PATH'], LOWBIT_FPMI_DATA_ROOT=str(tmp_path/'data'), MOCK_LOG=str(log), MOCK_COUNTER=str(counter))
    script=Path(__file__).resolve().parents[1]/'h200/prioritize_shared_distill_w2.sh'
    subprocess.run(['bash',str(script)],env=env,check=True,capture_output=True,text=True)
    lines=log.read_text().splitlines()
    assert any('afterany:23271:23272' in line for line in lines if line.startswith('sbatch'))
    a=next(i for i,x in enumerate(lines) if 'update JobId=23265 Dependency=afterok:1001' in x)
    b=next(i for i,x in enumerate(lines) if 'update JobId=23268 Dependency=afterok:1001' in x)
    cancelled=next(i for i,x in enumerate(lines) if x=='scancel 23264')
    assert a<cancelled and b<cancelled
    assert lines.index('scancel 22743') < lines.index('scancel 22742')
    assert lines.index('scancel 22746') < lines.index('scancel 22745')
    assert (w4/'after-shared-gate.txt').read_text().strip()=='1002'
