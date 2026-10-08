"""Independent metrics and paired article bootstrap from immutable predictions."""
import importlib
from pathlib import Path
import numpy as np
from research.kuki.data import FAMILIES, digest, write_json
from research.kuki.l1 import examples as l1_examples
from research.kuki.span_classifier import examples as l3_examples

E=importlib.import_module('research.stages.final.evaluate')
ROOT=E.ROOT
read=E.read

def macro(m):
    den=m.sum(-1)+m.sum(-2)
    return np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den>0).mean(-1)

def independent(docs,name,manifest):
    path=ROOT/'predictions/test'/name
    saved=read(path/'metrics.json')
    output={}
    if (path/'test-l1.jsonl').exists():
        rows=l1_examples(docs)
        pred=E.M.jsonl(path/'test-l1.jsonl')
        matrices={}
        for r,p in zip(rows,pred,strict=True):
            assert all(r[k]==p[k] for k in ['doc_id','start','end','annotation_index','target','weight'])
            m=matrices.setdefault(r['doc_id'],np.zeros((3,3)))
            m[r['target'],np.argmax(p['probabilities'])]+=r['weight']
        assert abs(macro(sum(matrices.values()))-saved['l1']['macro_f1'])<1e-12
        output['l1_f1']=matrices
    if (path/'test-l2.jsonl').exists():
        vals={}
        for d,r in zip(docs,E.M.jsonl(path/'test-l2.jsonl'),strict=True):
            assert d['doc_id']==r['doc_id']
            vals[d['doc_id']]=float(np.mean([(np.array(a['l2'])-r['probabilities'])**2 for a in d['annotations']]))
        assert abs(np.mean(list(vals.values()))-saved['l2']['brier'])<1e-12
        output['l2_brier']=vals
    if (path/'test-conditional.jsonl').exists():
        rows=l3_examples(docs)
        pred=E.M.jsonl(path/'test-conditional.jsonl')
        counts={}
        threshold=manifest['systems'][name]['conditional_threshold']
        for r,p in zip(rows,pred,strict=True):
            assert all(r[k]==p[k] for k in ['doc_id','start','end','annotation_index'])
            value=np.array(p['probabilities'])
            selected=value>=threshold
            selected[value.argmax()]=True
            gold=r['target'].astype(bool)
            v=counts.setdefault((r['doc_id'],r['annotation_index']),np.zeros(3))
            v+=np.array([(selected&gold).sum(),selected.sum(),gold.sum()])
        bydoc={}
        for (did,_),(tp,npred,ngold) in counts.items():
            bydoc.setdefault(did,[]).append(2*tp/(npred+ngold))
        vals={k:float(np.mean(v)) for k,v in bydoc.items()}
        assert abs(np.mean(list(vals.values()))-saved['conditional']['article_mean_f1'])<1e-12
        output['l3_conditional']=vals
        exact={}
        decoder=manifest['decoder']
        for d,r in zip(docs,E.M.jsonl(path/'test-chain.jsonl'),strict=True):
            assert d['doc_id']==r['doc_id']
            p=np.clip(np.array(r['probabilities']),1e-7,1-1e-7)
            scores=np.log(p/(1-p))+decoder['alpha']*np.array([v for _,_,v in r['pairs']])[:,None]
            ranked=sorted([(float(scores[i,j]),s,e,f) for i,(s,e,_) in enumerate(r['pairs']) for j,f in enumerate(FAMILIES)],key=lambda v:(-v[0],v[1],v[2],v[3]))
            predicted={(s,e,f) for v,s,e,f in ranked[:decoder['top_outputs']] if v>=decoder['threshold']}
            values=[]
            for a in d['annotations']:
                gold={(s['start'],s['end'],s['label']) for s in a['l3_persuasion']}
                values.append(2*len(gold&predicted)/max(1,len(gold)+len(predicted)))
            exact[d['doc_id']]=float(np.mean(values))
        assert abs(np.mean(list(exact.values()))-saved['chain_fixed']['exact_f1'])<1e-12
        output['l3_exact']=exact
    return output

def main():
    manifest=read(ROOT/'manifest.json')
    assert read(ROOT/'inference.json')['status']=='completed'
    E.check_frozen(manifest)
    docs=E.load_test()
    all_values={name:independent(docs,name,manifest) for name in manifest['systems']}
    for name in ['st4_control','st4_families']:
        for metric in ['l3_conditional','l3_exact']:
            all_values[name][metric]=all_values['st3'][metric]
    metrics={}
    draws={}
    for name,record in all_values.items():
        metrics[name]={}
        draws[name]={}
        for metric,values in record.items():
            keys=sorted(values)
            array=np.asarray([values[k] for k in keys])
            # Identical resampled article indices across models for each metric.
            ix=np.random.default_rng(42).integers(0,len(keys),(10000,len(keys)))
            if metric=='l1_f1':
                point=float(macro(array.sum(0)))
                sample=macro(array[ix].sum(1))
            else:
                point=float(array.mean())
                sample=array[ix].mean(1)
            draws[name][metric]=(keys,sample)
            metrics[name][metric]=dict(value=point,ci95=np.quantile(sample,[.025,.975]).tolist(),n_articles=len(keys))
    pairs=[('st3','l2','l2_brier'),('st3','l3','l3_conditional'),('st3','l3','l3_exact'),
           ('st6','l1','l1_f1'),('st6','l2','l2_brier'),
           ('st4_control','st3','l2_brier'),('st4_families','st3','l2_brier'),('st4_families','st4_control','l2_brier')]
    comparisons=[]
    for first,second,metric in pairs:
        ka,a=draws[first][metric];kb,b=draws[second][metric]
        assert ka==kb
        comparisons.append(dict(first=first,second=second,metric=metric,
            delta=metrics[first][metric]['value']-metrics[second][metric]['value'],
            ci95=np.quantile(a-b,[.025,.975]).tolist()))
    l1delta=metrics['st6']['l1_f1']['value']-metrics['l1']['l1_f1']['value']
    l2delta=metrics['st6']['l2_brier']['value']-metrics['l2']['l2_brier']['value']
    files=list((ROOT/'predictions').rglob('*.json*'))+[ROOT/'manifest.json',ROOT/'analyze.py',ROOT/'inference.json']
    write_json(ROOT/'results.json',dict(metrics=metrics,comparisons=comparisons,
        st6_criterion=l1delta>=-.01 and l2delta<=.003 and (l1delta>=.01 or l2delta<=-.003),
        test_articles=len(docs),test_answers=sum(len(d['annotations']) for d in docs),
        bootstrap_replicates=10000,bootstrap_seed=42,verification='independent primary metrics agree <1e-12',
        files={str(p):digest(p) for p in files}))
    write_json(ROOT/'article-statistics.json',{n:{m:{k:np.asarray(v).tolist() for k,v in vals.items()} for m,vals in record.items()} for n,record in all_values.items()})
    print({n:{m:round(v['value'],6) for m,v in record.items()} for n,record in metrics.items()})
    print(comparisons)

if __name__=='__main__':
    main()
