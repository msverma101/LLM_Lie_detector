"""Direct coordinate ablation. No optimizer, training, fitting, or label access."""
from pathlib import Path
import numpy as np

def edit(parent, strength=1., probe=None):
    candidate={k:np.array(v,copy=True) for k,v in parent.items()}
    if probe is None:
        candidate['W'][:,1780] *= 1-strength
    else:
        candidate['W'][probe,1780] *= 1-strength
    return candidate

if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    with np.load(root/'detector.npz',allow_pickle=False) as z:
        parent={k:z[k] for k in z.files}
    np.savez(root/'output'/'candidate.npz',**edit(parent))
