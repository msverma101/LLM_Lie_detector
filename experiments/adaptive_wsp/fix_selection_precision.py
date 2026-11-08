"""Apply the registered inclusive tie rule; retain all original result files."""
import argparse, json
from pathlib import Path
from study import better,sha,write
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();root=Path(a.run)
assert not (root/'FROZEN.json').exists() and not (root/'FINAL_OPENED.json').exists()
assert not (root/'selection_precision_correction.json').exists()
state=json.loads((root/'STATE.json').read_text());old_state=state.copy();all_feedback=[json.loads(p.read_text()) for p in sorted(root.glob('attempt_*/feedback.json'))]
# First attempt's parent is theta0; its preceding metric is stored in preparation log
# and the exact initial validation metric can be recovered from saved checkpoint scores.
from study import data,load,score,metrics
m,X=data(root.parent/'capture',['selection']);current='theta0.npz';vm=metrics(m,score(load(root/current),X));corrections=[]
for f in all_feedback:
    assert f['parent']==current,'a prior corrected decision would invalidate later parent attribution'
    keep=better(f['candidate_validation'],vm)
    corrections.append({'attempt':f['attempt'],'original_kept':f['kept'],'corrected_kept':keep,'feedback_sha256':sha(root/f"attempt_{f['attempt']:02d}/feedback.json")})
    if keep:current=f"attempt_{f['attempt']:02d}/wsp.npz";vm=f['candidate_validation']
state['incumbent']=current;state['incumbent_validation']=vm;state['precision_note']='selection_precision_correction.json'
c=root/'ordinary/WINNERS.json';old_controls=json.loads(c.read_text());new_controls={}
for arm in ['train_only','extra_data']:
    records=json.loads((root/f'ordinary/{arm}_all.json').read_text());best=records[0]
    for r in records[1:]:
        if better(r['validation'],best['validation']):best=r
    new_controls[arm]=best
write(root/'selection_precision_correction.json',{'old_state':old_state,'new_incumbent':current,'decisions':corrections,'old_control_winners':old_controls,'new_control_winners':new_controls,'tie_tolerance':1e-12})
write(root/'STATE.json',state);write(c,new_controls)
print('SELECTION_PRECISION_CORRECTED',json.dumps({'incumbent':current,'decisions':corrections}))
