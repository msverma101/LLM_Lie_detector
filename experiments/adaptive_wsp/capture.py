"""Fresh grouped last-token capture; no final score computation occurs here."""
import argparse, hashlib, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture_base import MODEL, MODEL_REV, DATA_REV, LAYERS, PARTS, QUADS, family, chat_render, digest
SEED = 20261108

class Union:
    def __init__(self): self.p = {}
    def find(self,x):
        p=self.p.setdefault(x,x)
        if p!=x: self.p[x]=self.find(p)
        return self.p[x]
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.p[max(a,b)]=min(a,b)

def aliases(r):
    msgs=json.loads(r.messages_json)
    first=next((m.get('content') for m in msgs if m.get('role')=='user'),r.messages_json)
    raw=json.dumps(first,sort_keys=True)
    norm=re.sub(r'\s+',' ',raw).strip().casefold()
    keys=['user:'+digest(norm),'exact:'+digest(r.messages_json+'\0'+str(r.completion))]
    old=[digest('fallback:'+raw)]
    pair='' if pd.isna(r.pair_id) else str(r.pair_id).strip()
    if pair:
        keys.append('pair:'+digest(pair)); old.append(digest('pair:'+pair))
        game=re.match(r'^(.*?\|Game \d+)\|',pair)
        if game: keys.append('episode:'+digest(game[1]))
    return keys,old

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--previous',required=True);a=ap.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'CAPTURE_DONE.json').exists(),'immutable completed capture'
    previous=pd.read_csv(a.previous).fillna(''); old_rows=set(previous.row_id); old_groups=set(previous.group_id)
    old_episodes=set()
    for pair in previous.pair_id:
        game=re.match(r'^(.*?\|Game \d+)\|',str(pair))
        if game: old_episodes.add('episode:'+digest(game[1]))
    frames=[];rejections={'char_prefilter':0,'unmapped_family':0,'examined_over_length':0,'examined_render_error':0}
    for quad in QUADS:
        p=hf_hub_download('AlignmentResearch/fibs-v1',f'data/{quad}-00000-of-00001.parquet',repo_type='dataset',revision=DATA_REV)
        d=pd.read_parquet(p,columns=['source','quadrant','pair_id','messages_json','completion','language','follow_up_category','human_verified'])
        d['dataset_split']=quad;d['dataset_row']=np.arange(len(d));d['family']=d.source.map(family)
        rejections['unmapped_family']+=int((d.family=='excluded_family').sum());d=d[d.family!='excluded_family'].copy()
        # Restriction is logged separately, including excluded non-ASCII contexts.
        long=(d.messages_json.str.len()+d.completion.fillna('').str.len())>2048*8
        rejections['char_prefilter']+=int(long.sum());d=d[~long].copy()
        d['row_id']=[digest(DATA_REV+':'+quad+':'+str(i)) for i in d.dataset_row]
        frames.append(d);print('LOADED',quad,len(d),flush=True)
    d=pd.concat(frames,ignore_index=True); uf=Union(); row_keys=[]; banned=set(old_episodes)
    for r in d.itertuples():
        keys,old=aliases(r)
        for k in keys[1:]: uf.join(keys[0],k)
        row_keys.append(keys)
        if r.row_id in old_rows or any(g in old_groups for g in old): banned.update(keys)
    banned_roots={uf.find(k) for k in banned}
    d['group_id']=[digest('component:'+uf.find(keys[0])) for keys in row_keys]
    banned_h={digest('component:'+k) for k in banned_roots};mask=d.group_id.isin(banned_h)
    rejections['old_connected_rows']=int(mask.sum());d=d[~mask].copy()
    mapping={f:p for p,fs in PARTS.items() for f in fs};d['partition']=d.family.map(mapping)
    clash=d.groupby('group_id').partition.nunique();bad=set(clash[clash>1].index)
    rejections['cross_family_partition_groups']=len(bad);d=d[~d.group_id.isin(bad)].copy()
    d['label']=d.quadrant.str.endswith('_deceptive').astype(int)
    target=d.partition=='target';cal=d.group_id.map(lambda g:int(digest(str(SEED)+g)[:12],16)%2==0)
    d.loc[target&cal,'partition']='calib';d.loc[target&~cal,'partition']='test'
    rejections['unused_calibration_positives']=int(((d.partition=='calib')&(d.label==1)).sum())
    d=d[~((d.partition=='calib')&(d.label==1))].copy()
    d['draw_order']=[digest(str(SEED)+g+rid) for g,rid in zip(d.group_id,d.row_id)]
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REV)
    chosen=[];rendered=[]
    for (part,fam,label),g in d.groupby(['partition','family','label'],sort=True):
        cap=100 if part in ['calib','test'] else (40 if part=='fit' else 50)
        # Alternate quadrants to balance when possible without oversampling sparse cells.
        queues=[list(x.sort_values('draw_order').index) for _,x in g.groupby('quadrant',sort=True)]
        order=[]
        while any(queues):
            for q in queues:
                if q: order.append(q.pop(0))
        kept=0
        for idx in order:
            if kept>=cap:break
            row=d.loc[idx]
            try: tokens=chat_render(tok,row)
            except Exception:rejections['examined_render_error']+=1;continue
            if len(tokens[0])>2048:rejections['examined_over_length']+=1;continue
            chosen.append(row.to_dict());rendered.append(tokens);kept+=1
        print('CELL',part,fam,label,kept,flush=True)
    meta=pd.DataFrame(chosen).reset_index(drop=True)
    assert meta.groupby('group_id').partition.nunique().max()==1
    assert not set(meta.row_id)&old_rows
    assert not meta.duplicated(['messages_json','completion']).any()
    meta['n_tokens']=[len(t[0]) for t in rendered];meta['answer_tokens']=[t[3]-t[1]+1 for t in rendered]
    for part,b in meta.groupby('partition'):b.to_json(root/f'cases_{part}.jsonl',orient='records',lines=True,force_ascii=False)
    meta.drop(columns=['messages_json','completion']).to_csv(root/'meta.csv',index=False)
    manifest={'client_date':'2026-11-08','seed':SEED,'model':MODEL,'model_revision':MODEL_REV,'dataset_revision':DATA_REV,'layers':LAYERS,'pool':'last_answer_token','max_tokens':2048,'dtype':'float16','previous_meta_sha256':hashlib.sha256(Path(a.previous).read_bytes()).hexdigest(),'rejections':rejections,'counts':meta.groupby(['partition','family','label']).size().rename('n').reset_index().to_dict('records'),'grouping':'union: raw pair, explicit run/game, normalized first user, exact transcript','historical_scope':'disjoint from recorded prior-pilot groups; all historical or semantic overlap not established','label_scope':'FIBS-labelled supplied final completions, no claim about monitored model intent','renderer':'copied previous text-only adaptation; not official EFC replay'}
    (root/'capture_manifest.json').write_text(json.dumps(manifest,indent=2));print('FRESH_SAMPLE',len(meta),json.dumps(rejections),flush=True)
    assert torch.cuda.is_available()
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=MODEL_REV,dtype=torch.float16).cuda().eval()
    assert not model.training
    start=time.monotonic();out=np.lib.format.open_memmap(root/'acts_last.npy',mode='w+',dtype=np.float16,shape=(len(meta),len(LAYERS),model.config.text_config.hidden_size if hasattr(model.config,'text_config') else model.config.hidden_size))
    for i,(ids,a0,t0,last) in enumerate(rendered):
        x=torch.tensor([ids],device='cuda')
        with torch.inference_mode():o=model(input_ids=x,output_hidden_states=True,use_cache=False,logits_to_keep=1)
        h=torch.stack([o.hidden_states[k+1][0,last] for k in LAYERS]).float().cpu().numpy().astype(np.float16)
        assert np.isfinite(h).all();out[i]=h
        del o,x,h
        if (i+1)%40==0:
            elapsed=time.monotonic()-start;print('CAPTURE',i+1,len(meta),'minutes',round(elapsed/60,2),'eta_minutes',round(elapsed/(i+1)*(len(meta)-i-1)/60,2),flush=True);out.flush()
    out.flush();del out,model;torch.cuda.empty_cache()
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    (root/'CAPTURE_DONE.json').write_text(json.dumps({'rows':len(meta),'minutes':(time.monotonic()-start)/60,'sha256':hashes},indent=2));print('CAPTURE_DONE',len(meta),flush=True)
if __name__=='__main__':main()
