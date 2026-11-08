"""Software checks only; generated examples are not detector research results."""
import contextlib, importlib.util, io, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import study

class HarnessTests(unittest.TestCase):
    def test_fit_replay_and_calibration_order(self):
        rng=np.random.default_rng(19);X=rng.normal(size=(90,6,9));y=(X[:,0,0]+X[:,2,1]>.0).astype(int);m=pd.DataFrame({'label':y,'family':'fixture'})
        t={'mu':X.mean(0),'sd':X.std(0),'W':np.zeros((6,9)),'b':np.zeros(6)}
        fitted,error=study.fit(t,m,X)
        self.assertLess(error,1e-10)
        self.assertTrue(np.array_equal(fitted['mu'],t['mu']))
        self.assertEqual(study.threshold(np.arange(600),.025),585.)
        self.assertEqual(study.threshold(np.arange(20),.025),float('inf'))
        self.assertTrue(study.better({'macro_recall':.3,'macro_auroc':.8},{'macro_recall':.2,'macro_auroc':.9}))
        self.assertFalse(study.better({'macro_recall':.2,'macro_auroc':.8},{'macro_recall':.3,'macro_auroc':.7}))

    def test_free_edit_cannot_change_frozen_scaler(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cap=p/'capture';run=p/'run';cap.mkdir();run.mkdir();rng=np.random.default_rng(3);X=rng.normal(size=(80,6,9));np.save(cap/'acts_last.npy',X)
            m=pd.DataFrame({'row_id':[str(i) for i in range(80)],'partition':['inspect']*40+['selection']*40,'label':[i%2 for i in range(80)],'family':['fixture']*80,'language':['en']*80,'group_id':[str(i) for i in range(80)]});m.to_csv(cap/'meta.csv',index=False)
            cases=m.iloc[:40].copy();cases['messages_json']='[{"role":"user","content":"fixture"}]';cases['completion']='fixture';cases.to_json(cap/'cases_inspect.jsonl',orient='records',lines=True)
            t={'mu':np.zeros((6,9)),'sd':np.ones((6,9)),'W':np.ones((6,9))*.1,'b':np.zeros(6)};study.save(run/'theta0.npz',t)
            study.write(run/'STATE.json',{'completed':0,'incumbent':'theta0.npz','incumbent_validation':{'macro_recall':.1,'macro_auroc':.5}})
            spec={'attempt':1,'parent':'theta0.npz',**{k:'fixture' for k in ['observation','hypothesis','prediction','control_prediction','edit_description','category']},'evidence_ids':['0']};study.write(p/'spec.json',spec)
            (p/'edit.py').write_text("def edit(t,context):\n    t['mu'][0,0]=1\n    return t\n")
            with self.assertRaisesRegex(AssertionError,'representation contract'):
                study.attempt(SimpleNamespace(run=str(run),capture=str(cap),spec=str(p/'spec.json'),edit=str(p/'edit.py')))
            self.assertEqual(json.loads((run/'STATE.json').read_text())['completed'],0)

if __name__=='__main__':unittest.main()
