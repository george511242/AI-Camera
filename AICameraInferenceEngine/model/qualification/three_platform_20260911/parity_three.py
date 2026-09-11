import os,sys,json,hashlib,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np,onnxruntime as ort
R=Path('/home/g2004/projects/VAC-AI-Camera-Source/vac-camera/vac-camera/AICameraInferenceEngine');O=R/'model/qualification/three_platform_20260911';sys.path.insert(0,str(R/'tools'))
kind=sys.argv[1];d=next(x for x in json.loads((O/'result.json').read_text()) if x['model']==kind)
size={'Age':112,'Gender':64,'Pose':224}[kind];rng=np.random.default_rng(20260911);x=rng.integers(0,256,(3,size,size,3),dtype=np.uint8).astype(np.float32);x[0]=0;x[1]=127
np.save(O/f'{kind.lower()}_qualification_inputs.npy',x)
if kind=='Age':
 from train_age_v4_mobilenetv3 import build_model
 m,_=build_model(np.linspace(-3,3,98).astype(np.float32),None);m.load_weights(d['source']);a=1+m(x,training=False).numpy().sum(axis=1,keepdims=True);inp=x.transpose(0,3,1,2);tol=.01
elif kind=='Gender':
 from export_ssrnet_gender_onnx import load_official_module
 mod=load_official_module(R/'model/versions/v3_age/training/vendor/SSR-Net/training_and_testing/SSRNET_model.py');m=mod.SSR_net_general(64,[3,3,3],1,1)();m.load_weights(d['source']);a=m(x,training=False).numpy();inp=x.transpose(0,3,1,2);tol=1e-4
else:
 import torch
 spec=importlib.util.spec_from_file_location('pose_network','/home/g2004/vendor/Lightweight-Head-Pose-Estimation/network/network.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);m=mod.Network(num_bins=66,M=99,bin_train=False);m.load_state_dict(torch.load(d['source'],map_location='cpu',weights_only=True));m.eval();inp=((x-np.array(d['mean'],np.float32))/np.array(d['std'],np.float32)).transpose(0,3,1,2)
 with torch.no_grad():a=np.concatenate([z.numpy().reshape(3,1) for z in m(torch.from_numpy(inp))],axis=1)
 tol=.01
so=ort.SessionOptions();so.intra_op_num_threads=2;s=ort.InferenceSession(d['onnx'],so,providers=['CPUExecutionProvider']);b=np.stack([np.concatenate([v.reshape(-1) for v in s.run(None,{s.get_inputs()[0].name:z[None]})]) for z in inp]);delta=float(np.max(np.abs(a-b)))
y=dict(model=kind,source_sha256=d['source_sha256'],onnx_sha256=d['onnx_sha256'],inputs='fixed synthetic zeros/midgray/seeded noise; no dataset or held-out labels',input_sha256=hashlib.sha256((O/f'{kind.lower()}_qualification_inputs.npy').read_bytes()).hexdigest(),source_outputs=a.tolist(),onnx_outputs=b.tolist(),max_absolute_tensor_delta=delta,decoded_output_delta=delta if kind!='Gender' else float(np.max((a>=.5)!=(b>=.5))),tolerance=tol,tolerance_basis='existing Age/Pose .01; Gender conservative supplemental scalar tolerance 1e-4',status='PASS' if np.isfinite(a).all() and np.isfinite(b).all() and delta<=tol else 'FAIL',optimizer_updates=0,dtype_layout='host NHWC FP32 transposed to ONNX NCHW FP32; Pose normalization as contract')
(O/f'{kind.lower()}_source_parity.json').write_text(json.dumps(y,indent=2));print(kind,y['status'],delta);assert y['status']=='PASS'
