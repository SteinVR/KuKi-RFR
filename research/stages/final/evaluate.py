"""One final inference series. No optimizer or threshold selection."""
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from research.kuki.data import digest, write_json, write_jsonl
from research.kuki.experiment import load_development
from research.kuki.global_pointer import GlobalPointer
from research.kuki.joint import JointModel
from research.kuki.frozen_message import FrozenResidual
from research.kuki.span_classifier import examples, score
from research.kuki.evaluation import score as l2_score
from research.kuki.train_l3 import configure, encode_documents

ROOT=Path('research/stages/final')
M=importlib.import_module('research.stages.ST3-l2-l3-joint.method')
R=importlib.import_module('research.stages.ST6-l1-l2-l3.method')
J=importlib.import_module('research.stages.ST6-l1-l2-l3.model')
E=importlib.import_module('research.stages.ST4-l3-to-l2.frozen.extract')
T=importlib.import_module('research.stages.ST2-l3-rhetoric.modular.train')
read=M.read

def check_frozen(manifest):
    for p,h in manifest['files'].items():
        assert digest(Path(p))==h,p

def payload(entry):
    return torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)['trainable']

def infer(part, split, manifest):
    dest=ROOT/'predictions'/split
    dest.mkdir(parents=True,exist_ok=False)
    systems=manifest['systems']
    pack=systems['st3']['packed']
    encoded=encode_documents(part,pack['encoder'],pack['config'])
    detrun=Path(manifest['detector']['run'])
    detector,_=M.CHAIN.load_model(detrun,GlobalPointer)
    candidates=json.loads(json.dumps(T.detector_predict(detector,part,encoded)))
    write_jsonl(dest/'detector.jsonl',candidates)
    del detector
    torch.cuda.empty_cache()
    if split=='dev':
        assert candidates==M.jsonl(detrun/'dev-predictions.jsonl')
    cc={r['doc_id']:r for r in candidates}
    result={}
    features={}
    for name in ['l2','l3','st3','l1','st6','st4_control','st4_families']:
        entry=systems[name]
        cfg=entry['packed']['config']
        configure(42)
        target=dest/name
        target.mkdir()
        if name.startswith('st4'):
            model=FrozenResidual(families=name=='st4_families').cuda()
            model.load_state_dict(payload(entry))
            model.eval()
            with torch.inference_mode():
                values=np.asarray([model(features[d['doc_id']]).sigmoid().cpu().numpy() for d in part])
            pred={'l2':values,'conditional':None,'chain':[]}
            M.persist_predictions(target,split,part,[],pred)
            metrics=dict(l2=l2_score(part,values),conditional=result['st3']['conditional'],chain_fixed=result['st3']['chain_fixed'])
        elif name in ['l1','st6']:
            if name=='l1':
                cfg=dict(cfg,l1_weight=1,l2_weight=0,l1_normalization=145/135)
            model=J.JointRoles(entry['packed']['encoder'],cfg).cuda()
            R.set_weights(model,payload(entry))
            pred=R.predict(model,part,encoded)
            metrics=R.persist(target,split,part,pred)
        else:
            model=JointModel(entry['packed']['encoder'],cfg).cuda()
            M.set_weights(model,payload(entry))
            rows=examples(part)
            pred=M.predict(model,part,encoded,rows,cc if cfg['l3_weight'] else None)
            M.persist_predictions(target,split,part,rows,pred)
            metrics={}
            if pred['l2'] is not None:
                metrics['l2']=l2_score(part,pred['l2'])
            if pred['conditional'] is not None:
                metrics['conditional']=score(rows,pred['conditional'],entry['conditional_threshold'])
                metrics['chain_fixed'],records,outputs=M.CHAIN.chain_score(part,pred['chain'],manifest['decoder']['alpha'],manifest['decoder']['threshold'])
                write_json(target/'chain-articles.json',records)
                write_jsonl(target/'chain-decoded.jsonl',outputs)
            if name=='st3':
                with torch.inference_mode():
                    for d in part:
                        did=d['doc_id']
                        features[did]=E.extract_item(model,encoded[did],cc[did]['pairs'],len(d['text']))
        if split=='dev':
            run=Path(entry['run'])
            if name in ['l1','st6']:
                for task in ['l1','l2']:
                    if pred.get(task) is not None:
                        file='dev-predictions.jsonl' if name=='l1' else f'dev-{task}.jsonl'
                        assert np.array_equal(pred[task],np.asarray([r['probabilities'] for r in M.jsonl(run/'best'/file)])),(name,task)
            else:
                for task in ['l2','conditional']:
                    if pred.get(task) is not None:
                        assert np.array_equal(pred[task],np.asarray([r['probabilities'] for r in M.jsonl(run/'best'/f'dev-{task}.jsonl')])),(name,task)
                if pred.get('chain'):
                    assert pred['chain']==M.jsonl(run/'best/dev-chain.jsonl'),name
        write_json(target/'metrics.json',metrics)
        result[name]=metrics
        del model
        torch.cuda.empty_cache()
        print(f'{split}: {name} inference and frozen replay passed',flush=True)
    write_json(dest/'metrics.json',result)
    return result

def load_test():
    dc=read('research/configs/data.json')
    manifest=read(dc['split_manifest'])
    assert digest(Path(dc['split_manifest']))==dc['split_sha256']
    assert digest(Path(dc['prepared_path']))==manifest['prepared_sha256']
    index={r['doc_id']:r for r in manifest['rows']}
    docs=[]
    for line in Path(dc['prepared_path']).read_text().splitlines():
        d=json.loads(line)
        row=index[d['doc_id']]
        if row['split']=='test':
            assert d['text_sha256']==row['text_sha256']
            d['group_id']=row['group_id']
            docs.append(d)
    docs.sort(key=lambda d:d['doc_id'])
    assert len(docs)==31 and len({d['group_id'] for d in docs})==31
    return docs

def main():
    started=time.perf_counter()
    manifest=read(ROOT/'manifest.json')
    check_frozen(manifest)
    assert not (ROOT/'test-opened.json').exists()
    configure(42)
    docs,_=load_development(read('research/configs/data.json'))
    infer(docs['dev'],'dev',manifest)
    write_json(ROOT/'dev-replay.json',dict(passed=True,systems=7,exact_probabilities=True))
    check_frozen(manifest)
    write_json(ROOT/'test-opened.json',dict(at=datetime.now(timezone.utc).isoformat(),manifest_sha256=digest(ROOT/'manifest.json')))
    test=load_test()
    infer(test,'test',manifest)
    check_frozen(manifest)
    write_json(ROOT/'inference.json',dict(status='completed',seconds=time.perf_counter()-started,systems=7,optimizer_steps=0,test_articles=len(test),manifest_sha256=digest(ROOT/'manifest.json')))

if __name__=='__main__':
    main()
