from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from edit import edit
S=Path('/Users/friedrichreichelt/Documents/Codex/2026-11-07/hast/work/wsp-method-skill/.agents/skills/detector-weight-space-debugging/scripts')
sys.path.insert(0,str(S));from audit_edit import inspect;from paired_eval import cutoff,counts,align,compare
p=Path(__file__).resolve().parent.parent;o=p/'output';m=pd.read_csv(o/'groups.csv');X=np.load(p/'activations.npy');z=np.load(p/'detector.npz');parent={k:z[k] for k in z.files};q=(X-parent['mu'])/parent['sd']
assert (o/'HYPOTHESIS.json').exists()
variants={'zero':edit(parent,0),'half':edit(parent,.5),'candidate':edit(parent,1),'reverse_half':edit(parent,1.5),'wrong_amplify':edit(parent,-1)}
for l in range(6):variants[f'probe_{l}']=edit(parent,1,l)
rng=np.random.default_rng(20261108);r=rng.normal(size=parent['W'].shape);r[:,1780]=0;r=r/np.linalg.norm(r,axis=1)[:,None]*abs(parent['W'][:,1780,None]);v={k:a.copy() for k,a in parent.items()};v['W']+=r;variants['random']=v
np.savez(o/'candidate.npz',**variants['candidate']);scores={};summary={}
for batch,(name,v) in enumerate(variants.items(),2):
 s=expit((q*v['W']).sum(-1)+v['b']).mean(-1);scores[name]=s;out=m.copy();out['score']=s;out.to_csv(o/f'scores_{name}.csv',index=False)
 with (o/'score_batches.jsonl').open('a') as f:f.write(json.dumps({'batch':batch,'checkpoint':name,'rows':len(m),'split':'all diagnostic groups','purpose':'preregistered strength/component/null check'})+'\n')
 cal=(m.analysis_split=='diagnostic_calibration')&(m.label==0);test=m.analysis_split=='transfer';t=cutoff(s[cal],.1)
 summary[name]={**counts(m.label[test],s[test],t),'threshold':t,'auroc':roc_auc_score(m.label[test],s[test])}
base=scores['zero'];new=scores['candidate'];pred=json.loads((o/'HYPOTHESIS.json').read_text())['predictions'];rows=[]
for r in pred:
 i=m.index[m.row_id==r['row_id']][0];rows.append({**r,'baseline_score':float(base[i]),'candidate_score':float(new[i]),'actual_mean_logit_delta':float(((q[i]*(variants['candidate']['W']-parent['W'])).sum(-1)).mean())})
(o/'prediction_check.json').write_text(json.dumps(rows,indent=2));(o/'control_metrics.json').write_text(json.dumps(summary,indent=2));audit=inspect(parent,variants['candidate']);(o/'edit_audit.json').write_text(json.dumps(audit,indent=2))
df=m[['row_id','group_id','family','label']].copy();a=df.assign(score=base);b=df.assign(score=new);cal=(m.analysis_split=='diagnostic_calibration')&(m.label==0);test=m.analysis_split=='transfer'
result=compare(align(a[cal],b[cal]),align(a[test],b[test]),alpha=.1,draws=1000);(o/'paired_development_eval.json').write_text(json.dumps(result,indent=2))
keep=result['recall_difference']>=.05 and result['joint_group_bootstrap']['recall_difference_95'][0]>0 and result['arms']['candidate']['fpr']-result['arms']['original']['fpr']<=.02 and result['gained_detections']>result['lost_detections'] and all(result['arms']['candidate']['per_family'][f]['recall']-result['arms']['original']['per_family'][f]['recall']>=-.1 for f in result['arms']['candidate']['per_family'])
(o/'DECISION.json').write_text(json.dumps({'decision':'provisional_development_repair' if keep else 'retain_baseline','keep_criteria_pass':bool(keep),'no_final_efficacy_claim':True,'candidate_sha256':hashlib.sha256((o/'candidate.npz').read_bytes()).hexdigest(),'score_batches':13,'reason':'Preregistered checks applied without selecting alternative strength; see paired evaluation and controls.','ordinary_fitting_control':'omitted because no classifier training/fitting authorized','uncorrupted_control':'not available in diagnostic packet; do not infer recovery of known fault'},indent=2))
print('CONTROL METRICS',json.dumps(summary,indent=2));print('PAIRED',json.dumps(result,indent=2));print('KEEP',keep)
