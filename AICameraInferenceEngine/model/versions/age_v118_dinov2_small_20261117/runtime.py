"""Fixed categorical median, equivalent to V70 global selected action."""
import numpy as np
def decode(logits):
 z=np.asarray(logits,dtype=np.float64);assert z.ndim in (1,2) and z.shape[-1]==101 and np.isfinite(z).all()
 p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);c=p.cumsum(-1);c[...,-1]=1.;return (c>=.5).argmax(-1).astype(np.float64)
