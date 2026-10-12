"""Matched OT/GGN saved-state compression with combined diagonal/subspace controls."""
import shutil
from pathlib import Path

import torch

from residual_curvature.config import load_config
from residual_curvature.data import dataset
from residual_curvature.models import Registry, make_model
from residual_curvature.provenance import checksum, new_run, write_json
from run_learnable_capacity import PARENT, TRAINING, FAMILIES, audit, read

FAMILIES_MATCHED=FAMILIES+[(f'combined_{f}_{r}',f'combined_{f}',r)
                          for f in ['teacher','state'] for r in [3,16]]
PROTOCOL='docs/ot_ggn_capacity_protocol_20261007.md'
ATTACHMENT=Path('/Users/georgtirpitz/.codex/attachments/d8de2a28-810d-4f48-be90-e235b8f645c7/Pasted text.txt')


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c=load_config(TRAINING/'resolved.yaml')
    directory=new_run(c)
    print(f'Matched capacity: {directory.resolve()}',flush=True)
    sources=read(directory/'provenance.json')['source_checksums']
    for name in [__file__, 'scripts/run_learnable_capacity.py','scripts/run_large_batch_oracle.py',
                 'scripts/run_mini_vit_allocation.py',PROTOCOL,'tests/test_combined_capacity.py','tests/test_capacity.py']:
        path=Path(name)
        if path.is_absolute():
            path=path.relative_to(Path.cwd())
        sources[str(path)]=checksum(path)
    for name,digest in sources.items():
        out=directory/'source'/name
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,out)
        assert checksum(out)==digest
    write_json(directory/'source_archive_checksums.json',sources)
    shutil.copyfile(ATTACHMENT,directory/'supplied_critique.txt')
    write_json(directory/'protocol.json',{'document':PROTOCOL,'parent':str(PARENT),'families':FAMILIES_MATCHED,
               'targets':['ggn','ot'],'training_launched':False,'attachment_sha256':checksum(ATTACHMENT)})
    x,y,manifest=dataset(c)
    assert manifest==read(TRAINING/'split_manifest.json')
    write_json(directory/'split_manifest.json',manifest)
    model=make_model(c)
    registry=Registry(model)
    rows,operators,states,parents=[],[],[],{}
    for trajectory in ['searched_soap','exact_tensor_ggn']:
        for step in [10,25,50]:
            source=PARENT/f'{trajectory}-{step:04d}'
            parents[str(source/'arrays.pt')]=checksum(source/'arrays.pt')
            write_json(directory/'parent_arrays.json',parents)
            for target in ['ggn','ot']:
                key=f'{target}-{source.name}'
                try:
                    r,o,s=audit(source,directory/key,c,x,y,registry,model,'damping',
                                target=target,families=FAMILIES_MATCHED,common_factor_rhs=True)
                    rows.extend(r);operators.extend(o);states.append(s)
                    write_json(directory/'all_candidates.json',rows)
                    write_json(directory/'all_operators.json',operators)
                    write_json(directory/'all_states.json',states)
                except Exception as error:
                    write_json(directory/'failure.json',{'state':key,'error':repr(error)})
                    raise
    assert len(rows)==420 and len(operators)==204 and len(states)==12
    write_json(directory/'complete.json',{'completed':True,'candidate_steps':420,'operator_states':204,
                                        'states':6,'targets':2,'training_launched':False})
    print(f'Completed: {directory.resolve()}',flush=True)


if __name__=='__main__':
    main()
