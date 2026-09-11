exec(open('/tmp/audit_three.py').read().split('result=[]')[0])
import subprocess,sys,importlib.metadata
records=json.loads((O/'audit.json').read_text())
for d in records:
 for platform,t in d['targets'].items():
  if not t['job_required']:continue
  dest=Path(t['path']);log=O/f"{d['model'].lower()}_{platform}.log"
  assert not dest.exists() and not log.exists()
  cmd=[sys.executable,str(R/'tools/convert_rknn_fp16.py'),'--source',d['onnx'],'--output',str(dest),'--target',platform,'--mean',*map(str,d['mean']),'--std',*map(str,d['std'])]
  t.update(command=cmd,log=str(log),source_onnx_sha256=d['onnx_sha256'],toolkit_version=importlib.metadata.version('rknn-toolkit2'))
  assert sha(Path(d['onnx']))==d['onnx_sha256']
  with log.open('w') as f:ret=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=O).returncode
  txt=log.read_text();t.update(status='PASS' if ret==0 and dest.exists() else 'FAIL',returncode=ret,warnings_errors=[l for l in txt.splitlines() if 'warning' in l.lower() or l.startswith('E ') or 'error' in l.lower()],unsupported_replaced_operators='See full conversion log; no manual operator replacements')
  if dest.exists():t.update(sha256=sha(dest),bytes=dest.stat().st_size,embedded_metadata=embedded(dest))
  assert sha(Path(d['onnx']))==d['onnx_sha256']
  (O/'result.json').write_text(json.dumps(records,indent=2));print(d['model'],platform,t['status'],flush=True)
  if t['status']=='FAIL':break
