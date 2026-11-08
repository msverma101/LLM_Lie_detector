"""Create complete portable evidence, including raw activations, without changing results."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
assert (root/'run/FINAL_DONE.json').exists()
archive=root/'adaptive_wsp_complete.tar.xz'
with tarfile.open(archive,'w:xz',preset=3) as t:
    for name in ['capture','run']:
        t.add(root/name,arcname=name)
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in ['.py','.md','.json','.csv','.log']:
            t.add(p,arcname=p.name)
b=archive.read_bytes();chunk=6*1024*1024;parts=[]
for i,start in enumerate(range(0,len(b),chunk)):
    part=root/f'transfer_{i:03d}.bin';part.write_bytes(b[start:start+chunk]);parts.append({'name':part.name,'bytes':part.stat().st_size,'sha256':hashlib.sha256(part.read_bytes()).hexdigest()})
manifest={'file':archive.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'parts':parts,'includes_raw_activations':True}
(root/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('COMPLETE_EXPORT_READY',json.dumps(manifest),flush=True)
