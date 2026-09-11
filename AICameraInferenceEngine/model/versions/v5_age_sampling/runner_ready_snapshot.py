#!/usr/bin/env python3
"""Fixed Age v5 experiment. Explicit ordered stages; never overwrite a training run."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'model/versions/v5_age_sampling'
CACHE = OUT / 'cache_corrected'
SPLITS = OUT / 'split_repair'
SOURCE = Path('/home/g2004/datasets/vac-distance/source/utkface/crop_part1')
CHECKPOINT = ROOT / 'model/versions/v4_age_mobilenetv3/experiments/V4/age_v4_best.weights.h5'
CHECKPOINT_SHA = '9f3e9850cb4f07d20cbb0ee90ddc87574c4ee00c74a9eb4b3a8c0d5af2033272'
SEED = 20260905
VARIANTS = {'160px': .4, '80px': .35, '40px': .25}
WEIGHTS = {'0-17': 1., '18-54': 1.5, '55-80': 2., '81-99': 1.}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False))


def handoff(stage, message):
    p = ROOT / 'HANDOFF.md'
    p.write_text(f'## Age v5 approved experiment — {stage}\n{message}\nArtifacts: model/versions/v5_age_sampling/. CPU only; seed20260905; maximum3 epochs/arm; no other model changes or Phase1/frozen confirmation.\n\n---\n\n' + p.read_text())


def group(age):
    return '0-17' if age <= 17 else '18-54' if age <= 54 else '55-80' if age <= 80 else '81-99'


def cache_index(split):
    assert split in ('train', 'validation')
    return SPLITS / f'{split}_cache.jsonl'


def arm_config(arm):
    assert arm in ('control', 'treatment')
    return dict(train_manifest=str(SPLITS/'train.jsonl'),
                validation_manifest=str(SPLITS/'validation.jsonl'),
                train_cache_index=str(cache_index('train')),
                validation_cache_index=str(cache_index('validation')),
                cache_root=str(CACHE), checkpoint=str(CHECKPOINT),
                checkpoint_sha256=CHECKPOINT_SHA, seed=SEED,
                recipe=json.loads((OUT/'run_config.json').read_text()),
                sampling_weights=WEIGHTS if arm=='treatment' else {g:1. for g in WEIGHTS})


def sampling_probabilities(rows, arm):
    weights=arm_config(arm)['sampling_weights']
    totals={v:sum(weights[group(r['age'])] for r in rows if r['variant']==v) for v in VARIANTS}
    p=np.array([VARIANTS[r['variant']]*weights[group(r['age'])]/totals[r['variant']] for r in rows])
    assert np.isclose(p.sum(),1)
    return p


def check_ready():
    # Approved repair gate only; never rerun the historical leakage audit.
    gate=json.loads((SPLITS/'result.json').read_text())
    assert gate['status']=='PASS_WITH_RESIDUAL_LIMITATIONS'
    provenance=json.loads((SPLITS/'provenance.json').read_text())
    required=[SPLITS/n for n in ['train.jsonl','validation.jsonl','train_cache.jsonl',
                               'validation_cache.jsonl','source_family_groups.jsonl','result.json']]
    for path in required:
        assert sha(path)==provenance['outputs'][str(path)]
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    counts={}
    for split, expected in [('train',(5712,17037)),('validation',(1300,3846))]:
        sources=read(SPLITS/f'{split}.jsonl'); rows=read(cache_index(split))
        by_id={r['source_id']:r for r in sources}
        assert len(by_id)==len(sources)
        assert {r['source_id'] for r in rows}==set(by_id)
        assert len({(r['source_id'],r['height']) for r in rows})==len(rows)==3*len(sources)
        heights={160,80,40} if split=='train' else {115,57,38}
        for r in rows:
            source=by_id[r['source_id']]
            assert r['height'] in heights and r['variant']==f"{r['height']}px"
            assert r['age']==source['age'] and r['source_sha256']==source['content_sha256']
            if r['detected']:
                path=(CACHE/r['file']).resolve()
                assert path.is_relative_to(CACHE.resolve()) and sha(path)==r['crop_sha256']
        counts[split]=dict(sources=len(sources),rows=len(rows),detected=sum(r['detected'] for r in rows))
        assert (len(sources),counts[split]['detected'])==expected
    return counts


def metrics(rows, pred):
    result={}
    for height in ['overall',115,57,38]:
        result[str(height)]={}
        for g in ['overall','0-17','18-54','55-80','81-99']:
            idx=[i for i,r in enumerate(rows) if (height=='overall' or r['height']==height) and (g=='overall' or group(r['age'])==g)]
            valid=[i for i in idx if rows[i]['detected']]
            errors=np.array([pred[i]-rows[i]['age'] for i in valid],float)
            result[str(height)][g]=dict(total=len(idx),detected=len(valid),mae=float(np.abs(errors).mean()),
                within7=float((np.abs(errors)<=7).sum()/len(idx)),conditional_within7=float((np.abs(errors)<=7).mean()),signed=float(errors.mean()))
    return result


def init_model():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    from train_age_v4_mobilenetv3 import build_model, compile_model, initial_cutpoints
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.utils.set_random_seed(SEED);tf.config.experimental.enable_op_determinism()
    model,backbone=build_model(initial_cutpoints(read(SPLITS/'train.jsonl'),1,99),None)
    model.load_weights(CHECKPOINT)
    backbone.trainable=True
    for layer in backbone.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
    compile_model(model,1e-4)
    digest=hashlib.sha256()
    for w in model.weights: digest.update(str(w.shape).encode());digest.update(w.numpy().tobytes())
    return model,digest.hexdigest()


def evaluate(model, rows):
    import tensorflow as tf
    from train_age_v4_mobilenetv3 import decode_cached
    valid=[r for r in rows if r['detected']]
    ds=tf.data.Dataset.from_tensor_slices(([str(CACHE/r['file']) for r in valid],[r['age'] for r in valid]))
    ds=ds.map(lambda path,age:decode_cached(path,age,False)[0],num_parallel_calls=4,deterministic=True).batch(64).prefetch(2)
    p=model.predict(ds,verbose=0)
    assert np.isfinite(p).all() and (np.diff(p,axis=1)<=1e-6).all(), 'ordinal/finite failure'
    values=1+p.sum(axis=1);pred=np.full(len(rows),np.nan);pred[[i for i,r in enumerate(rows) if r['detected']]]=values
    return pred


def weight_fingerprint(model):
    digest=hashlib.sha256()
    for w in model.weights:
        digest.update(str(w.shape).encode());digest.update(w.numpy().tobytes())
    return digest.hexdigest()


def dry_run():
    assert not (OUT/'dry_run.json').exists(), 'Read the completed dry run; do not repeat.'
    assert not (OUT/'initialization.json').exists()
    assert not (OUT/'control').exists() and not (OUT/'treatment').exists()
    counts=check_ready()
    configs={arm:arm_config(arm) for arm in ['control','treatment']}
    differences=[k for k in configs['control'] if configs['control'][k]!=configs['treatment'][k]]
    assert differences==['sampling_weights']
    rows=[r for r in read(cache_index('train')) if r['detected']]
    probabilities={arm:sampling_probabilities(rows,arm) for arm in configs}
    assert not np.array_equal(probabilities['control'],probabilities['treatment'])
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    from unittest.mock import patch
    fingerprints={}; steps={}
    # Fail closed if any training/update entry point is accidentally introduced.
    def forbidden(*args,**kwargs):
        raise AssertionError('Optimizer/training operations forbidden during dry run')
    with patch.object(tf.keras.Model,'fit',forbidden), patch.object(tf.keras.Model,'train_on_batch',forbidden), patch.object(tf.keras.optimizers.Adam,'apply_gradients',forbidden):
        for arm in configs:
            tf.keras.backend.clear_session()
            model,fingerprint=init_model()
            assert int(model.optimizer.iterations)==0
            assert weight_fingerprint(model)==fingerprint
            fingerprints[arm]=fingerprint; steps[arm]=int(model.optimizer.iterations)
    assert fingerprints['control']==fingerprints['treatment']
    code_paths=[Path(__file__),ROOT/'tools/train_age_v4_mobilenetv3.py',ROOT/'tools/train_ssrnet_age_v32_ordinal.py',ROOT/'inference/utils/img_utils.py',OUT/'run_config.json',SPLITS/'provenance.json']
    immutable={str(p):sha(p) for p in code_paths}
    write(OUT/'initialization.json',dict(checkpoint_sha256=CHECKPOINT_SHA,**fingerprints,optimizer_steps=0,immutable_code=immutable,arm_configs=configs))
    result=dict(status='READY',counts=counts,arms=configs,only_arm_difference=differences,
                weight_fingerprints=fingerprints,optimizer_iterations=steps,optimizer_updates=0,
                cache_hashes_match=True,original_manifests_or_indexes_used=False,
                sampling_probability_sha256={arm:hashlib.sha256(p.tobytes()).hexdigest() for arm,p in probabilities.items()},
                immutable_code=immutable,residual_limitations=json.loads((SPLITS/'result.json').read_text())['residual_limitations'],
                next_action='STOP after READY; no training in this step. Await authorization to run CONTROL, then TREATMENT.')
    write(OUT/'dry_run.json',result)
    command=f'CUDA_VISIBLE_DEVICES=-1 /home/g2004/miniforge3/envs/vac-age-v3/bin/python {Path(__file__).resolve()}'
    handoff('RUNNER READY — NO TRAINING; STOP',
        'Latest authorization: runner wiring and no-train dry run only; leakage PASS accepted within documented scope. '
        'Runner: '+str(Path(__file__).resolve())+'; saved result: '+str(OUT/'dry_run.json')+'. '
        'BOTH arms use '+str(SPLITS/'train.jsonl')+', '+str(cache_index('train'))+', '+str(SPLITS/'validation.jsonl')+', '+str(cache_index('validation'))+'. '
        'Crop root '+str(CACHE)+' (unchanged files); checkpoint '+str(CHECKPOINT)+' SHA256 '+CHECKPOINT_SHA+'. '
        'Dry-run counts: train5712 sources/17037 successful crops; validation1300/3846. Loaded model fingerprints identical: '+fingerprints['control']+'. '
        'Only sampling differs; optimizer updates0; control0/treatment0. All repaired membership/provenance/crop hashes match. No old split/index read, no leakage rerun/cache rebuild. '
        'Bootstrap uses connected source-family group IDs. Unknown cross-dataset derivative provenance remains incomplete; fixed v4 historical exposure cannot be undone. '
        'Original audit FAIL retained; active gate split_repair/result.json. Initialization and READY bind code/config/provenance hashes; do not repeat completed dry run. '
        '\nExact CONTROL command (after authorization): `'+command+' --stage control`'
        '\nExact TREATMENT command (only after CONTROL completes): `'+command+' --stage treatment`'
        '\nEXACT NEXT ACTION: STOP and report READY. No training authorized in this step; subsequent CONTROL then TREATMENT require user continuation. Fixed seed20260905, same v4 recipe, max3epochs each, no Phase1/frozen confirmation.')
    print(json.dumps(result,indent=2),flush=True)


def train(arm):
    check_ready();init=json.loads((OUT/'initialization.json').read_text())
    assert json.loads((OUT/'dry_run.json').read_text())['status']=='READY'
    assert init['arm_configs'][arm]==arm_config(arm)
    assert all(sha(p)==h for p,h in init['immutable_code'].items())
    if arm=='treatment': assert (OUT/'control/result.json').exists()
    dest=OUT/arm; assert not dest.exists(), 'Never restart/overwrite a started arm.'
    dest.mkdir();write(dest/'started.json',dict(time=time.time(),arm=arm))
    model,fingerprint=init_model();assert fingerprint==init[arm]
    import tensorflow as tf
    from train_age_v4_mobilenetv3 import decode_cached
    trainrows=[r for r in read(cache_index('train')) if r['detected']];val=read(cache_index('validation'))
    rng=np.random.default_rng(SEED);p=sampling_probabilities(trainrows,arm)
    indices=rng.choice(len(trainrows),size=3*253*64,replace=True,p=p)
    np.save(dest/'draw_indices.npy',indices)
    write(dest/'draws.json',dict(by_group=dict(Counter(group(trainrows[i]['age']) for i in indices)),by_variant=dict(Counter(trainrows[i]['variant'] for i in indices)),initial_fingerprint=fingerprint,checkpoint_sha256=CHECKPOINT_SHA,seed=SEED))
    history=[];start=time.monotonic()
    class Finite(tf.keras.callbacks.Callback):
        def on_train_batch_end(self,batch,logs=None):
            if not all(np.isfinite(v) for v in (logs or {}).values()):raise RuntimeError('Nonfinite training metric')
    for epoch in range(1,4):
        draw=[trainrows[i] for i in indices[(epoch-1)*253*64:epoch*253*64]]
        ds=tf.data.Dataset.from_tensor_slices(([str(CACHE/r['file']) for r in draw],[r['age'] for r in draw]))
        ds=ds.map(lambda path,age:decode_cached(path,age,True),num_parallel_calls=4,deterministic=True).batch(64,drop_remainder=True).prefetch(2)
        h=model.fit(ds,epochs=1,verbose=2,callbacks=[Finite()])
        assert int(model.optimizer.iterations)==epoch*253
        pred=evaluate(model,val);np.save(dest/f'epoch{epoch}_predictions.npy',pred)
        m=metrics(val,pred);write(dest/f'epoch{epoch}_metrics.json',m)
        model.save_weights(dest/f'epoch{epoch}.weights.h5')
        history.append(dict(epoch=epoch,fit=h.history,metrics=m,elapsed=time.monotonic()-start))
        write(dest/'history.json',history)
    check_ready()
    write(dest/'result.json',dict(epochs=3,optimizer_steps=759,initial_fingerprint=fingerprint,checkpoint_sha256=CHECKPOINT_SHA,elapsed=time.monotonic()-start,final=history[-1]['metrics']))
    handoff(arm.upper()+' COMPLETE', f'{arm}:3 epochs/759steps complete; result and raw validation predictions saved. Next '+('--stage treatment.' if arm=='control' else '--stage compare. No Phase1/frozen run.'))


def bootstrap(rows, control, treatment, height, g):
    rr=[(i,r) for i,r in enumerate(rows) if (height=='overall' or r['height']==height) and group(r['age'])==g]
    families={m['source_id']:g['group_id'] for g in read(SPLITS/'source_family_groups.jsonl') for m in g['members'] if m['split']=='validation'}
    ids=sorted({families[r['source_id']] for _,r in rr});z=np.zeros((len(ids),5));index={s:i for i,s in enumerate(ids)}
    for i,r in rr:
        v=z[index[families[r['source_id']]]];v[0]+=1
        if r['detected']:
            a=control[i]-r['age'];b=treatment[i]-r['age'];v[1]+=1;v[2]+=abs(b)-abs(a);v[3]+=int(abs(b)<=7)-int(abs(a)<=7);v[4]+=b-a
    rng=np.random.default_rng(SEED);samples=z[rng.integers(0,len(ids),(1000,len(ids)))].sum(axis=1);point=z.sum(axis=0)
    result={}
    for name,col,den in [('mae',2,1),('within7',3,0),('signed',4,1)]:
        valid=samples[:,den]>0
        values=samples[valid,col]/samples[valid,den]
        result[name]=dict(delta=float(point[col]/point[den]),ci95=np.quantile(values,[.025,.975]).tolist(),valid_bootstrap_replicates=int(valid.sum()))
    return result


def compare():
    assert (OUT/'control/result.json').exists() and (OUT/'treatment/result.json').exists()
    assert not (OUT/'comparison.json').exists()
    rows=read(cache_index('validation'));control=np.load(OUT/'control/epoch3_predictions.npy');cm=metrics(rows,control);results=[]
    for epoch in [1,2,3]:
        treatment=np.load(OUT/f'treatment/epoch{epoch}_predictions.npy');tm=metrics(rows,treatment);dif={};gates=[]
        for height in ['overall',115,57,38]:
            dif[str(height)]={g:bootstrap(rows,control,treatment,height,g) for g in ['0-17','18-54','55-80']}
            c,a,e=[dif[str(height)][g] for g in ['0-17','18-54','55-80']]
            gates.extend([c['within7']['delta']>=-.01,c['mae']['delta']<=.3,a['within7']['delta']>=.03,a['mae']['delta']<=0,e['within7']['delta']>=.05,e['mae']['delta']<=-1,e['within7']['delta']>max(c['within7']['delta'],a['within7']['delta'])])
        results.append(dict(epoch=epoch,pass_gates=all(gates),treatment_metrics=tm,delta_ci=dif))
    passing=[r for r in results if r['pass_gates']]
    best=max(passing,key=lambda r:(r['delta_ci']['overall']['55-80']['within7']['delta'],-r['treatment_metrics']['overall']['55-80']['mae'])) if passing else results[-1]
    result=dict(status='PASS' if passing else 'FAIL',control_epoch=3,selected_treatment_epoch=best['epoch'] if passing else None,reported_treatment_epoch=best['epoch'],control_metrics=cm,reported=best,all_epochs=results,bootstrap='1000 paired connected source-family resamples,percentile95%; repeated sizes kept together; E2E within7,conditional MAE/signed',next_action='STOP; report before Phase1 confirmation' if passing else 'STOP; no weight/LR/seed/v5.1 retry')
    if passing:
        import shutil
        target=OUT/'frozen_age_v5.weights.h5';shutil.copyfile(OUT/f"treatment/epoch{best['epoch']}.weights.h5",target);result['frozen_sha256']=sha(target)
    write(OUT/'comparison.json',result)
    handoff('TREATMENT '+result['status']+'; STOP', f"Validation gates {result['status']}. See comparison.json for group/size metrics and treatment-control source-bootstrap CIs. {result['next_action']}. No external confirmation executed.")
    print(result['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['dry-run','control','treatment','compare'],required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage in ['control','treatment']:train(args.stage)
    else:globals()[args.stage.replace('-', '_')]()
