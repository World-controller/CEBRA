from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cebra

caes, controls, baos, metrics = cebra.engines()
EXAMPLE = ROOT / 'examples/smoke/test.jsonl'
CONFIG = ROOT / 'configs/hotpotqa.json'

def run(*args, ok=True):
    p = subprocess.run([sys.executable, str(ROOT/'src/cebra.py'), *map(str,args)], capture_output=True, text=True, cwd=ROOT)
    if ok:
        assert p.returncode == 0, p.stderr + p.stdout
    else:
        assert p.returncode != 0, p.stdout
    return p

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def dumped(path, values):
    path.write_text(''.join(json.dumps(v)+'\n' for v in values))
    return path

@pytest.mark.parametrize('command', ['', 'train','calibrate','retain','evaluate'])
def test_help_without_site_packages(command):
    p=subprocess.run([sys.executable,'-S',str(ROOT/'src/cebra.py'),*([command] if command else []),'--help'],capture_output=True,text=True)
    assert p.returncode == 0
    if not command:
        assert 'CEBRA' in p.stdout

@pytest.mark.parametrize('dataset', ['hotpotqa','2wiki','musique'])
def test_frozen_artifacts(dataset):
    cfg=cebra.read_json(ROOT/f'configs/{dataset}.json')
    model_path=(ROOT/'configs'/cfg['model']).resolve()
    metadata=cebra.read_json(cebra.model_metadata_path(model_path))
    model,names=cebra.load_checked_model(model_path,metadata,caes)
    assert names==caes.FEATURE_NAMES
    assert cebra.sha256(model_path)==cfg['model_sha256']
    assert model.named_steps['model'].C==1
    assert cfg['operating_point']['min_keep_per_iter']==0

@pytest.mark.parametrize('smoke', [True,False])
def test_retain_exact_parity_and_input_unchanged(tmp_path,smoke):
    original=EXAMPLE.read_bytes()
    output=tmp_path/'out.jsonl'
    run('retain','--config',CONFIG,'--fpath',EXAMPLE,'--output',output,*(['--smoke-example'] if smoke else []))
    cfg=cebra.read_json(CONFIG); mp=(CONFIG.parent/cfg['model']).resolve()
    model,names=caes.load_model(str(mp)); nlp=cebra.load_nlp(smoke)
    source=rows(EXAMPLE); before=deepcopy(source)
    expected=[caes.replay_sample_calibrated(deepcopy(row),nlp=nlp,model=model,feature_names=names,threshold=.7,min_keep_per_iter=0,debug_features=False) for row in source]
    assert rows(output)==expected
    assert source==before and EXAMPLE.read_bytes()==original
    meta=cebra.read_json(Path(str(output)+'.meta.json'))
    assert meta['synthetic_smoke_example'] is True
    assert meta['regex_smoke_mode']==smoke


def test_metrics_exact_legacy_parity(tmp_path):
    output=tmp_path/'retained.jsonl'
    run('retain','--config',CONFIG,'--fpath',EXAMPLE,'--output',output,'--smoke-example')
    new=tmp_path/'new.json'; old=tmp_path/'old.json'
    run('evaluate','--fpath',output,'--output',new)
    subprocess.run([sys.executable,str(ROOT/'src/evaluation/cmrg_retrieve_metrics.py'),'--fpath',str(output),'--output',str(old)],check=True,capture_output=True)
    assert cebra.read_json(new)==cebra.read_json(old)
    assert cebra.read_json(new)['empty_final_context_count']==1


def test_all_four_commands_and_calibration_parity(tmp_path):
    model=tmp_path/'custom.pkl'; lock=tmp_path/'lock.json'
    run('train','--fpath',ROOT/'examples/smoke/train.jsonl','--output',model,'--smoke-example')
    run('calibrate','--model',model,'--fpath',ROOT/'examples/smoke/calibration.jsonl','--split-manifest',ROOT/'examples/smoke/calibration.manifest.json','--output',lock,'--smoke-example')
    result=cebra.read_json(lock)
    baseline=rows(ROOT/'examples/smoke/calibration.jsonl')
    budget=tmp_path/'budget.jsonl'
    dumped(budget,[controls.budget_only_replay(deepcopy(r),3) for r in baseline])
    m,names=caes.load_model(str(model))
    expected=baos.sweep_for_budget_match(ROOT/'examples/smoke/calibration.jsonl',budget,nlp=None,model=m,feature_names=names,thresholds=[i/10 for i in range(1,10)],min_keep_values=[0,1],max_recall_drop=.01,sweep_limit=3)
    assert result['selected']==expected['selected']
    assert result['candidates']==expected['candidates']
    assert len(result['candidates'])==18 and result['test_data_used_for_selection'] is False
    retained=tmp_path/'retained.jsonl'
    run('retain','--model',model,'--lock',lock,'--fpath',EXAMPLE,'--output',retained,'--smoke-example')
    run('evaluate','--fpath',retained,'--output',tmp_path/'metrics.json')
    overlap=run('retain','--model',model,'--lock',lock,'--fpath',ROOT/'examples/smoke/calibration.jsonl','--output',tmp_path/'bad.jsonl','--smoke-example',ok=False)
    assert 'overlaps calibration' in overlap.stderr
    formal=run('retain','--model',model,'--lock',lock,'--fpath',EXAMPLE,'--output',tmp_path/'bad2.jsonl',ok=False)
    assert 'smoke-trained' in formal.stderr


def candidate(t,k,r=.8,e=0,minimum=0):
    return {'threshold':t,'min_keep_per_iter':minimum,'metrics':{'final_context_k':k,'recall':r,'empty_final_context_count':e}}

@pytest.mark.parametrize('candidates',[
    [candidate(.2,3),candidate(.1,3)],
    [candidate(.1,4,.8,2),candidate(.2,3,.8,1)],
    [candidate(.1,4,.1,2),candidate(.2,3,.2,1,1)],
    [candidate(.1,4,.8,0,1),candidate(.2,3,.8,1,0)],
])
def test_baos_tie_and_fallback_unchanged(candidates):
    budget={'final_context_k':3,'recall':.8,'empty_final_context_count':0}
    original=baos.choose_candidate(candidates,budget_metrics=budget,max_recall_drop=.01)
    assert original==baos.choose_candidate(list(reversed(candidates)),budget_metrics=budget,max_recall_drop=.01)
    assert isinstance(original['relaxed_empty_gate'],bool)
    assert cebra.engines()[2].choose_candidate is baos.choose_candidate


def test_terminates_duplicates_and_gold_independence(tmp_path):
    source=rows(EXAMPLE)
    for row in source:row['oracle']=['deliberately-not-the-gold']
    altered=dumped(tmp_path/'gold-changed.jsonl',source)
    a=tmp_path/'a.jsonl';b=tmp_path/'b.jsonl'
    for inp,out in [(EXAMPLE,a),(altered,b)]:run('retain','--config',CONFIG,'--fpath',inp,'--output',out,'--smoke-example')
    for left,right in zip(rows(a),rows(b)):
        for k in [key for key in left if key.isdigit()]:
            assert left[k]==right[k]
            assert all(not d['keep_for_qa'] for d in left[k]['docs'] if d['label']=='<TERMINATE>')
        chunks,_,_=caes.extract_final_chunks(left)
        assert len({d['id'] for d in chunks})==len(chunks)


def test_stale_annotations_removed_on_copy(tmp_path):
    source=rows(EXAMPLE)
    for row in source:
        for chunk in baos.iter_docs(row):chunk.update(keep_for_qa=True,retention_score=123,cmrg_features={'bogus':1})
    before=deepcopy(source); cleaned=cebra.clean_rows(source,baos)
    assert source==before
    assert all(not set(chunk)&baos.REPLAY_FIELDS for row in cleaned for chunk in baos.iter_docs(row))

@pytest.mark.parametrize('bad', ['wrong-hash','wrong-split','wrong-ids','wrong-count'])
def test_calibration_manifest_rejected(tmp_path,bad):
    manifest=cebra.read_json(ROOT/'examples/smoke/calibration.manifest.json')
    field={'wrong-hash':'sha256','wrong-split':'split','wrong-ids':'question_ids','wrong-count':'samples'}[bad]
    manifest[field]='incorrect';p=tmp_path/'manifest.json';cebra.write_json(p,manifest)
    out=tmp_path/'out.json'
    run('calibrate','--config',CONFIG,'--fpath',ROOT/'examples/smoke/calibration.jsonl','--split-manifest',p,'--output',out,'--smoke-example',ok=False)
    assert not out.exists()


def test_missing_nlp_fails_without_fallback(monkeypatch):
    import spacy
    def missing(_):raise OSError('missing')
    monkeypatch.setattr(spacy,'load',missing)
    with pytest.raises(ValueError,match='no fallback'):cebra.load_nlp(False)
    assert cebra.load_nlp(True) is None


def test_wrong_nlp_version_rejected(monkeypatch):
    import spacy
    monkeypatch.setattr(spacy,'__version__','0.0.0')
    with pytest.raises(ValueError,match='requires spaCy'):cebra.load_nlp(False)


def test_model_hash_and_feature_mismatch(tmp_path):
    path=ROOT/'models/caes-2wiki-train.pkl';meta=cebra.read_json(cebra.model_metadata_path(path))
    wrong=dict(meta,model_sha256='0'*64)
    with pytest.raises(ValueError,match='SHA-256'):cebra.load_checked_model(path,wrong,caes)
    a=pickle.loads(path.read_bytes());a['feature_names']=list(reversed(a['feature_names']))
    p=tmp_path/'bad.pkl';p.write_bytes(pickle.dumps(a));meta['model_sha256']=cebra.sha256(p)
    with pytest.raises(SystemExit,match='Feature mismatch'):cebra.load_checked_model(p,meta,caes)

@pytest.mark.parametrize('case',['unmarked','empty-file','five-rounds','gap','bad-label','missing-id'])
def test_input_boundaries(tmp_path,case):
    data=rows(EXAMPLE)
    if case=='unmarked':data[0].pop('_cebra_example')
    if case=='empty-file':data=[]
    if case=='five-rounds':data[0].update({str(i):{'docs':[]} for i in range(2,5)})
    if case=='gap':data[0]['3']=data[0].pop('1')
    if case=='bad-label':data[0]['0']['docs'][0]['label']='unknown'
    if case=='missing-id':data[0]['0']['docs'][0].pop('id')
    p=dumped(tmp_path/'input.jsonl',data)
    with pytest.raises(ValueError):cebra.load_rows(p,smoke=True)


def test_no_overwrite(tmp_path):
    p=tmp_path/'data';p.write_text('unchanged')
    with pytest.raises(ValueError):cebra.new_output(p,[p])
    with pytest.raises(ValueError):cebra.new_output(p,[])
    assert p.read_text()=='unchanged'


def test_nlp_override_restored_after_exception():
    before=caes.load_nlp
    with pytest.raises(RuntimeError):
        with cebra.nlp_override(caes,None):raise RuntimeError('test')
    assert caes.load_nlp is before


def test_train_exact_legacy_parameter_parity(tmp_path):
    import argparse
    import numpy as np
    path=ROOT/'examples/smoke/train.jsonl'
    new=tmp_path/'new.pkl';old=tmp_path/'old.pkl'
    run('train','--fpath',path,'--output',new,'--smoke-example')
    args=argparse.Namespace(fpath=str(path),output=str(old),summary_output=None,train_offset=0,train_limit=4,max_iter=1000,seed=42,drop_features='')
    with cebra.nlp_override(caes,None):
        caes.train_model(args)
    a=pickle.loads(new.read_bytes());b=pickle.loads(old.read_bytes())
    assert a['summary']==b['summary']
    for step,attributes in [('model',['coef_','intercept_']),('scaler',['mean_','scale_'])]:
        for attribute in attributes:
            np.testing.assert_array_equal(getattr(a['pipeline'].named_steps[step],attribute),getattr(b['pipeline'].named_steps[step],attribute))


def test_calibration_rejects_training_overlap_before_scoring(tmp_path):
    from argparse import Namespace
    path=ROOT/'examples/smoke/calibration.jsonl'
    args=Namespace(fpath=path,split_manifest=ROOT/'examples/smoke/calibration.manifest.json')
    with pytest.raises(ValueError,match='overlaps'):
        cebra.check_calibration_manifest(args,rows(path),{'training_ids':['synthetic-11']})


def test_evaluation_requires_gold(tmp_path):
    data=rows(EXAMPLE)
    for row in data:row.pop('oracle')
    path=dumped(tmp_path/'no-gold.jsonl',data)
    result=run('evaluate','--fpath',path,'--output',tmp_path/'metrics.json',ok=False)
    assert 'requires oracle' in result.stderr
