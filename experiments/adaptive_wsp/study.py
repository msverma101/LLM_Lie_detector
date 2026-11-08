"""Fixed-representation adaptive WSP harness; explicit free edit modules are separate."""
import argparse, copy, hashlib, importlib.util, json, math, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
SEED=20261108
ALPHA=.025
CS=[.0001,.0003,.001,.003,.01,.03,.1,.3,1,3]
COUNTS={'score_calls':0,'score_rows':0}

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,t):np.savez_compressed(p,**t)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def threshold(neg,alpha=ALPHA):
    v=np.sort(neg);r=math.ceil((len(v)+1)*(1-alpha));return float(v[r-1]) if r<=len(v) else float('inf')
def score(t,X):
    COUNTS['score_calls']+=1;COUNTS['score_rows']+=len(X)
    z=np.einsum('nld,ld->nl',(X-t['mu'])/t['sd'],t['W'])+t['b']
    return expit(z).mean(1)
def logits(t,X):return np.einsum('nld,ld->nl',(X-t['mu'])/t['sd'],t['W'])+t['b']
def metrics(m,s,cut=None):
    y=m.label.to_numpy();f=m.family.to_numpy();auc={str(k):float(roc_auc_score(y[f==k],s[f==k])) for k in sorted(set(f)) if len(set(y[f==k]))==2}
    if cut is None:cut=threshold(s[y==0])
    pred=s>cut;pos=y==1;neg=~pos
    fr={str(k):float(pred[(f==k)&pos].mean()) for k in sorted(set(f)) if np.any((f==k)&pos)}
    return {'tp':int(np.sum(pred&pos)),'fn':int(np.sum(~pred&pos)),'fp':int(np.sum(pred&neg)),'tn':int(np.sum(~pred&neg)),'recall':float(pred[pos].mean()) if pos.any() else None,'fpr':float(pred[neg].mean()) if neg.any() else None,'macro_recall':float(np.mean(list(fr.values()))) if fr else None,'macro_auroc':float(np.mean(list(auc.values()))) if auc else None,'per_family_auroc':auc,'per_family_recall':fr}
def data(root,names):
    m=pd.read_csv(Path(root)/'meta.csv').fillna('');mask=m.partition.isin(names).to_numpy();X=np.load(Path(root)/'acts_last.npy',mmap_mode='r')[mask].astype(np.float64)
    return m[mask].reset_index(drop=True),X

def fit(t,m,X,C=.01):
    out=copy.deepcopy(t);ref=[]
    for l in range(X.shape[1]):
        x=(X[:,l]-t['mu'][l])/t['sd'][l];c=LogisticRegression(C=C,max_iter=3000,random_state=SEED).fit(x,m.label)
        assert c.n_iter_[0]<3000
        out['W'][l]=c.coef_[0];out['b'][l]=c.intercept_[0];ref.append(c.predict_proba(x)[:,1])
    err=float(np.max(np.abs(score(out,X)-np.mean(ref,axis=0))));assert err<1e-8
    return out,err

def better(new,old):
    d=new['macro_recall']-old['macro_recall']
    return d>.005+1e-12 or (abs(d)<=.005+1e-12 and new['macro_auroc']>old['macro_auroc']+1e-12)

def details(root,t,m,X,tag,ids=None,full=True):
    s=score(t,X);cut=threshold(s[m.label==0]);p=s>cut;y=m.label.to_numpy();z=logits(t,X)
    if ids is None:
        ids=[]
        for mask in [(y==0)&p,(y==1)&~p,(y==0)&~p,(y==1)&p]:
            # Highest false alarms; lowest misses; near-threshold correct controls.
            ix=np.flatnonzero(mask)
            if len(ix):
                order=ix[np.argsort(np.abs(s[ix]-cut))]
                ids.extend(m.iloc[order[:3]].row_id.tolist())
    cases=[]
    for rid in ids:
        ix=np.flatnonzero(m.row_id.to_numpy()==rid)
        if not len(ix):continue
        i=ix[0]; x=(X[i]-t['mu'])/t['sd'];c=x*t['W'];layers=[]
        for l in range(len(c)):
            top=np.argsort(np.abs(c[l]))[-8:][::-1]
            layers.append({'layer_position':l,'logit':float(z[i,l]),'top':[{'j':int(j),'contribution':float(c[l,j]),'w':float(t['W'][l,j]),'x_standardized':float(x[l,j])} for j in top]})
        row=m.iloc[i].to_dict();row.update(score=float(s[i]),temporary_threshold=cut,diagnostic_error='FP' if y[i]==0 and p[i] else 'FN' if y[i]==1 and not p[i] else 'correct',layers=layers)
        cases.append(row)
    summary={'metrics':metrics(m,s),'weight_norms':np.linalg.norm(t['W'],axis=1).tolist(),'biases':t['b'].tolist(),'max_abs_w':np.max(np.abs(t['W']),axis=1).tolist(),'layer_auroc':[float(roc_auc_score(y,z[:,l])) for l in range(len(t['W']))],'case_ids':ids,'by_language':{str(k):metrics(m.iloc[np.flatnonzero(m.language.to_numpy()==k)].reset_index(drop=True),s[m.language.to_numpy()==k],cut) for k in sorted(set(m.language)) if np.sum(m.language==k)>=10}}
    write(Path(root)/(tag+'_cases.json'),cases);write(Path(root)/(tag+'_summary.json'),summary)
    out=m.copy();out['score']=s;out['temporary_threshold']=cut;out['prediction']=p.astype(int);out.to_csv(Path(root)/(tag+'_predictions.csv'),index=False)
    return summary,cases

def prepare(a):
    root=Path(a.run);root.mkdir(parents=True,exist_ok=True);assert not (root/'STATE.json').exists()
    m,X=data(a.capture,['fit']);mu=X.mean(0);sd=X.std(0);sd[sd<1e-6]=1
    t={'mu':mu,'sd':sd,'W':np.zeros_like(mu),'b':np.zeros(X.shape[1])};t,err=fit(t,m,X)
    save(root/'theta0.npz',t)
    md,Xd=data(a.capture,['inspect']);raw=pd.read_json(Path(a.capture)/'cases_inspect.jsonl',lines=True).fillna('');md=md.merge(raw[['row_id','messages_json','completion']],on='row_id',validate='one_to_one')
    summary,cases=details(root,t,md,Xd,'initial')
    mv,Xv=data(a.capture,['selection']);v=metrics(mv,score(t,Xv))
    write(root/'STATE.json',{'created_unix':time.time(),'client_date':'2026-11-08','completed':0,'incumbent':'theta0.npz','incumbent_validation':v,'capture_done_sha256':sha(Path(a.capture)/'CAPTURE_DONE.json'),'theta0_sha256':sha(root/'theta0.npz'),'sklearn_replay_error':err,'frozen_contract':{'mu_sha256':hashlib.sha256(mu.tobytes()).hexdigest(),'sd_sha256':hashlib.sha256(sd.tobytes()).hexdigest(),'shape':list(t['W'].shape),'pool':'last_answer_token','equal_layer_weights':True},'prediction_counts':COUNTS.copy()})
    print('PREPARED',json.dumps({'fit_n':len(m),'diagnostic_n':len(md),'validation_n':len(mv),'diagnostic':summary,'validation':v,'case_ids':[r['row_id'] for r in cases]}),flush=True)

def attempt(a):
    root=Path(a.run);assert not (root/'FROZEN.json').exists();assert not (root/'FINAL_OPENED.json').exists()
    state=json.loads((root/'STATE.json').read_text());n=state['completed']+1;assert n<=10
    spec=json.loads(Path(a.spec).read_text());assert spec['attempt']==n and spec['parent']==state['incumbent']
    for k in ['observation','hypothesis','prediction','control_prediction','evidence_ids','edit_description','category']:assert spec.get(k),k
    folder=root/f'attempt_{n:02d}';folder.mkdir(exist_ok=False)
    write(folder/'spec.json',spec);(folder/'edit.py').write_bytes(Path(a.edit).read_bytes())
    parent=load(root/spec['parent']);base=load(root/'theta0.npz')
    md,Xd=data(a.capture,['inspect']);raw=pd.read_json(Path(a.capture)/'cases_inspect.jsonl',lines=True).fillna('');md=md.merge(raw[['row_id','messages_json','completion']],on='row_id',validate='one_to_one')
    module_spec=importlib.util.spec_from_file_location('free_edit_'+str(n),folder/'edit.py');module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    # The saved module is authored and reviewed by the agent, not a hidden optimizer.
    t=module.edit(copy.deepcopy(parent),{'X':Xd,'meta':md.copy()})
    assert set(t)==set(parent)
    for k in ['mu','sd']:assert np.array_equal(t[k],base[k]),'representation contract changed'
    for k in t:assert t[k].shape==base[k].shape and np.isfinite(t[k]).all()
    delta={k:float(np.linalg.norm(t[k]-parent[k])) for k in ['W','b']};assert sum(delta.values())>1e-12,'real nonzero coefficient edit required'
    save(folder/'wsp.npz',t)
    ids=list(dict.fromkeys(spec['evidence_ids']+json.loads((root/'initial_summary.json').read_text())['case_ids']))
    ds,cases=details(folder,t,md,Xd,'after',ids)
    before_s=score(parent,Xd);after_s=score(t,Xd);changes=[]
    for rid in ids:
        ii=np.flatnonzero(md.row_id.to_numpy()==rid)
        if len(ii):i=ii[0];changes.append({'row_id':rid,'label':int(md.iloc[i].label),'before':float(before_s[i]),'after':float(after_s[i]),'delta':float(after_s[i]-before_s[i])})
    mv,Xv=data(a.capture,['selection']);vm=metrics(mv,score(t,Xv));keep=better(vm,state['incumbent_validation'])
    rng=np.random.default_rng(SEED+n);rand_metrics=[]
    for j in range(20):
        rt=copy.deepcopy(parent)
        for k in ['W','b']:
            if k=='W':
                for l in range(len(rt[k])):
                    d=t[k][l]-parent[k][l];noise=rng.normal(size=d.shape);rt[k][l]+=noise/np.linalg.norm(noise)*np.linalg.norm(d)
            else:
                d=t[k]-parent[k];noise=rng.normal(size=d.shape);rt[k]+=noise/np.linalg.norm(noise)*np.linalg.norm(d)
        save(folder/f'random_{j:02d}.npz',rt);rand_metrics.append(metrics(mv,score(rt,Xv)))
    write(folder/'random_feedback.json',rand_metrics)
    previous=state['incumbent'];state['completed']=n
    if keep:state['incumbent']=str(Path(folder.name)/'wsp.npz');state['incumbent_validation']=vm
    feedback={'attempt':n,'parent':previous,'candidate_validation':vm,'candidate_diagnostic':ds['metrics'],'coefficient_delta_l2':delta,'case_changes':changes,'kept':keep,'incumbent':state['incumbent'],'incumbent_validation':state['incumbent_validation'],'random_control_feedback':'saved, withheld until freeze','prediction_counts':COUNTS.copy()}
    write(folder/'feedback.json',feedback);write(root/'STATE.json',state);print('ADAPTIVE_FEEDBACK',json.dumps(feedback),flush=True)

def controls(a):
    root=Path(a.run);assert not (root/'FINAL_OPENED.json').exists();folder=root/'ordinary';folder.mkdir(exist_ok=True);base=load(root/'theta0.npz');mv,Xv=data(a.capture,['selection'])
    winners={}
    for arm,parts in [('train_only',['fit']),('extra_data',['fit','inspect'])]:
        m,X=data(a.capture,parts);records=[]
        for i,C in enumerate(CS):
            t,err=fit(base,m,X,C);p=folder/f'{arm}_{i:02d}.npz';save(p,t);vm=metrics(mv,score(t,Xv));records.append({'path':str(p.relative_to(root)),'C':C,'validation':vm,'fit_rows':len(m),'replay_error':err})
        best=records[0]
        for rec in records[1:]:
            if better(rec['validation'],best['validation']):best=rec
        winners[arm]=best;write(folder/(arm+'_all.json'),records)
    write(folder/'WINNERS.json',winners);print('CONTROLS_READY: two arms, ten C choices each; feedback withheld',flush=True)

def freeze(a):
    root=Path(a.run);state=json.loads((root/'STATE.json').read_text());assert state['completed']==10;assert not (root/'FROZEN.json').exists()
    c=json.loads((root/'ordinary/WINNERS.json').read_text());winners={'original':'theta0.npz','wsp':state['incumbent'],'ordinary_train':c['train_only']['path'],'ordinary_extra':c['extra_data']['path']}
    if state['incumbent']!='theta0.npz':
        folder=Path(state['incumbent']).parent;spec=json.loads((root/folder/'spec.json').read_text());winners['wsp_parent']=spec['parent']
        for j in range(20):winners[f'random_{j:02d}']=str(folder/f'random_{j:02d}.npz')
    # Supplementary attribution controls, fixed from the frozen winner, never selected.
    selected=load(root/winners['wsp']);base=load(root/'theta0.npz');confidence=copy.deepcopy(base);coordinates=copy.deepcopy(selected)
    factors=[]
    for l in range(len(base['W'])):
        valid=np.abs(base['W'][l])>1e-10;factor=float(np.median(selected['W'][l,valid]/base['W'][l,valid]));assert factor>0
        factors.append(factor);confidence['W'][l]*=factor;confidence['b'][l]*=factor;coordinates['W'][l]/=factor;coordinates['b'][l]/=factor
    save(root/'ablation_confidence_only.npz',confidence);save(root/'ablation_coordinates_only.npz',coordinates)
    winners['ablation_confidence_only']='ablation_confidence_only.npz';winners['ablation_coordinates_only']='ablation_coordinates_only.npz'
    write(root/'ABLATION_CONTRACT.json',{'row_factors':factors,'status':'supplementary, fixed after validation selection before test; no winner reselection'})
    out={'client_date':'2026-11-08','created_unix':time.time(),'winners':winners,'sha256':{k:sha(root/p) for k,p in winners.items()},'alpha':ALPHA,'protocol_sha256':sha(Path(__file__).parent/'PROTOCOL.md'),'study_sha256':sha(__file__),'prediction_counts':COUNTS.copy()};write(root/'FROZEN.json',out);print('FROZEN',json.dumps(out),flush=True)

def sample_groups(m,rng):
    groups=m.group_id.unique();chosen=rng.choice(groups,len(groups),replace=True);mapping={g:np.flatnonzero(m.group_id.to_numpy()==g) for g in groups};return np.concatenate([mapping[g] for g in chosen])
def final(a):
    root=Path(a.run);assert not (root/'FINAL_OPENED.json').exists(),'final was already opened';f=json.loads((root/'FROZEN.json').read_text())
    for k,p in f['winners'].items():assert sha(root/p)==f['sha256'][k]
    write(root/'FINAL_OPENED.json',{'created_unix':time.time(),'frozen_sha256':sha(root/'FROZEN.json')})
    mc,Xc=data(a.capture,['calib']);mt,Xt=data(a.capture,['test']);assert (mc.label==0).all();scores={};out={};cutoffs={}
    for k,p in f['winners'].items():
        t=load(root/p);sc,st=score(t,Xc),score(t,Xt);scores[k]=(sc,st);cut=threshold(sc);cutoffs[k]=cut;v=metrics(mt,st,cut)
        n=v['fp']+v['tn'];v['fpr_upper_95']=float(beta.ppf(.95,v['fp']+1,n-v['fp'])) if v['fp']<n else 1.;v['threshold']=cut;v['secondary_alpha_05']=metrics(mt,st,threshold(sc,.05));out[k]=v
        for part,m,s in [('calib',mc,sc),('test',mt,st)]:
            tab=m.copy();tab['score']=s;tab['threshold']=cut;tab['prediction']=(s>cut).astype(int);tab.to_csv(root/f'final_{part}_{k}.csv',index=False)
    arms=['original','wsp','ordinary_train','ordinary_extra'];rng=np.random.default_rng(SEED);boots={k:[] for k in arms};diffs={k:[] for k in arms if k!='wsp'}
    for _ in range(2000):
        ci=sample_groups(mc,rng);ti=sample_groups(mt,rng);m=mt.iloc[ti].reset_index(drop=True)
        vals={k:metrics(m,scores[k][1][ti],threshold(scores[k][0][ci])) for k in arms}
        for k in arms:boots[k].append([vals[k]['recall'],vals[k]['fpr']])
        for k in diffs:diffs[k].append([vals['wsp']['recall']-vals[k]['recall'],vals['wsp']['fpr']-vals[k]['fpr']])
    for k in arms:out[k]['bootstrap_95_recall_fpr']=np.quantile(boots[k],[.025,.975],axis=0).T.tolist()
    intervals={k:np.quantile(v,[.025,.975],axis=0).T.tolist() for k,v in diffs.items()}
    result={'client_date':'2026-11-08','calibration_rows':len(mc),'calibration_groups':mc.group_id.nunique(),'test_rows':len(mt),'test_groups':mt.group_id.nunique(),'alpha':ALPHA,'arms':out,'wsp_minus_comparator_95_recall_fpr':intervals,'bootstrap_draws':2000,'bootstrap_scope':'paired source groups, including calibration; conditional on familiar families','prediction_counts':COUNTS.copy()};write(root/'FINAL_DONE.json',result);write(root/'thresholds.json',cutoffs);print('FINAL_RESULT',json.dumps(result),flush=True)

def replay(a):
    root=Path(a.run);done=json.loads((root/'FINAL_DONE.json').read_text());m,X=data(a.capture,['test']);mc,Xc=data(a.capture,['calib']);f=json.loads((root/'FROZEN.json').read_text())
    for k,p in f['winners'].items():
        t=load(root/p);s=score(t,X);sc=score(t,Xc);saved=pd.read_csv(root/f'final_test_{k}.csv');assert np.max(np.abs(s-saved.score.to_numpy()))<1e-10
        v=metrics(m,s,threshold(sc));assert all(v[n]==done['arms'][k][n] for n in ['tp','fn','fp','tn'])
    print('ACTIVATION_AND_SCORE_REPLAY_OK',len(f['winners']),'arms',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','attempt','controls','freeze','final','replay']);ap.add_argument('--capture',required=True);ap.add_argument('--run',required=True);ap.add_argument('--spec');ap.add_argument('--edit');a=ap.parse_args();globals()[a.phase](a)
