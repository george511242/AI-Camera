import json,hashlib,re,ast
from pathlib import Path
import onnx
R=Path('/home/g2004/projects/VAC-AI-Camera-Source/vac-camera/vac-camera/AICameraInferenceEngine');O=R/'model/qualification/three_platform_20260911';O.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def embedded(p):
 for v in re.findall(rb'[ -~]{20,}',p.read_bytes()):
  if v.startswith(b"{'attrs'"):
   try:return ast.literal_eval(v.decode())
   except (ValueError,SyntaxError):pass
 return {}
models=[('YuNet','v1_baseline','yunet_n_640_640',None,[0,0,0],[1,1,1],'BGR raw 0..255 NHWC host; square pad640; corrected bgr_xywh; 12 outputs cls/obj/bbox/kps strides8/16/32, score.75 NMS.3'),('Age','v4_age_mobilenetv3','age_v4_mobilenetv3_large_112','experiments/V4/age_v4_best.weights.h5',[0,0,0],[1,1,1],'RGB raw float32 NHWC host112; margin.45; built-in -1..1; scalar age already decoded 1+sum(98 probabilities)'),('Gender','v2_finetuned','gender_ssrnet_v2_64x64','training/runs/gender_v2_seed20260905/gender_ssrnet_v2_selected.weights.h5',[0,0,0],[1,1,1],'BGR raw NHWC host64; margin.45; scalar >=.5 male'),('Pose','pose_v2_runtime','head_pose_v2_e1_224x224','training/runs/seed20260909/head_pose_v2_epoch1.pkl',[123.675,116.28,103.53],[58.395,57.12,57.375],'RGB raw float32 NHWC host224; margin.6; RKNN mean/std supplies ImageNet normalization; outputs roll,yaw,pitch degrees')]
result=[]
for name,ver,stem,source,mean,std,contract in models:
 base=R/'model/versions'/ver;op=base/'onnx'/f'{stem}.onnx';g=onnx.load(str(op));onnx.checker.check_model(g)
 def io(v):return dict(name=v.name,shape=[d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim],dtype=onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type))
 d=dict(model=name,version=ver,onnx=str(op),onnx_sha256=sha(op),onnx_checker='PASS',opset=[x.version for x in g.opset_import],inputs=[io(v) for v in g.graph.input],outputs=[io(v) for v in g.graph.output],source=str(base/source) if source else str(op),source_kind='checkpoint' if source else 'native canonical ONNX; upstream framework checkpoint not available',contract=contract,mean=mean,std=std,targets={})
 d['source_exists']=Path(d['source']).is_file();d['source_sha256']=sha(Path(d['source'])) if d['source_exists'] else None
 for platform in ['rk3566','rk3576','rk3588']:
  old=base/'rknn'/platform/f'{stem}_fp16.rknn';t=dict(target_platform=platform,old_path=str(old),exists=old.exists(),runtime='PENDING',physical_runtime_performed=False)
  if old.exists():
   t.update(sha256=sha(old),embedded_metadata=embedded(old));t['status']='EXISTING BUT UNVERIFIED';t['reason']='No immutable source ONNX SHA256-to-artifact build record established'
   if name=='Gender':
    rec=base/'validation/gender_v2_conversion.json';c=json.loads(rec.read_text());a=c['targets'][platform]
    assert c['source_onnx']['sha256']==d['onnx_sha256'] and a['sha256']==t['sha256']
    t.update(status='PASS',reason='source/artifact SHA256 matches saved conversion record',conversion_record=str(rec),toolkit_version=c['toolkit_version'],log=None,warnings=c['warnings'])
   if name=='Pose':
    rec=base/'deployment/conversion.json';c=json.loads(rec.read_text());par=base/'deployment/onnx_parity.json';q=json.loads(par.read_text());assert c[platform]['sha256']==t['sha256'] and q['onnx_sha256']==d['onnx_sha256'] and q['checkpoint_sha256']==d['source_sha256']
    t.update(status='PASS',reason='locked deployment record plus ONNX parity hash chain',conversion_record=str(rec),toolkit_version=c['toolkit_version'],log=str(base/'deployment'/f'convert_{platform}.log'),warnings=c[platform]['warnings'])
  else:t.update(status='MISSING',reason='No artifact')
  t['job_required']=t['status']!='PASS';t['path']=str(base/'rknn'/platform/f'{stem}_fp16_qualified_20260911.rknn') if old.exists() and t['job_required'] else str(old)
  d['targets'][platform]=t
 result.append(d)
(O/'audit.json').write_text(json.dumps(result,indent=2));print([(d['model'],{k:v['status'] for k,v in d['targets'].items()}) for d in result])
