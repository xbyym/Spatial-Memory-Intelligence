"""Shared inference planning and execution for individual inputs and batches."""
import argparse,copy,hashlib,json,os,subprocess,sys,time,signal,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import build_opener,ProxyHandler

HY_QWEN='hy15'
WAN_QWEN='wan22'
ROOT=Path(__file__).resolve().parents[1]
STOP=threading.Event()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def add(args,**kw):
 for k,v in kw.items():args += ['--'+k,str(v).lower()if isinstance(v,bool)else str(v)]
 return args
def absolute_inputs(row,cases):
 row=copy.deepcopy(row)
 for k in ['image_path','pose_json_path','trajectory_npz_path']:
  if k in row and not Path(row[k]).is_absolute():row[k]=str(cases/row[k])
 return row

def input_fingerprints(parameters):
 """Bind a plan to actual input bytes, not only reusable filenames."""
 result={}
 for field in ['image_path','pose_json_path','trajectory_npz_path']:
  if parameters.get(field):
   path=Path(parameters[field]);data=path.read_bytes()
   result[field]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 return result

def save_plan(p,execute=False,resume=False):
 """Never let planning or a changed input overwrite an existing run's evidence."""
 destination=Path(p['output']);existing=destination/'launch_plan.json'
 if existing.exists():
  if read(existing)!=p:raise ValueError('Existing plan, input bytes, or source changed. Use a new --output')
  if execute and not resume:raise ValueError('Existing run: use --resume or a new --output')
  if execute and (destination/'failed.json').exists():raise ValueError('Failed attempt is preserved. Retry in a new --output')
  return
 if any((destination/name).exists()for name in ['finished.json','failed.json','running.json']):
  raise ValueError('Existing run evidence has no launch plan. Use a new --output')
 write(existing,p)

def completed_for_resume(p):
 path=Path(p['output'])/'finished.json'
 if not path.exists():return False
 receipt=read(path)
 if receipt.get('completed')is not True or not receipt.get('media') or receipt.get('source_sha256')!=p['source_sha256']:
  raise ValueError('Invalid completion receipt; preserve this attempt and use a new --output')
 return True
def load_models(path):
 path=Path(path).expanduser().resolve();models=read(path)
 def resolve(value):
  p=Path(value).expanduser()
  return str((p if p.is_absolute()else path.parent/p).resolve())
 for key in ['hy15_backbone','hy15_action','wan22_backbone','wan22_transformer','wan22_action']:
  models[key]=resolve(models[key])
 models['qwen_checkpoints']={key:resolve(value)for key,value in models['qwen_checkpoints'].items()}
 return models
def plan(case_dir,method,models,clone,code,out,port,gpu_group,*,geometry_seed=0,geometry_points=None,input_data=None,backbone=None,environment=None,expected_frames=None):
 if input_data is None or backbone not in ('HY1.5','Wan2.2'):
  raise ValueError('Provide a HY1.5 or Wan2.2 input JSON')
 env=dict(environment or {});job=out;case_name=str(input_data.get('name',case_dir.name))
 smi=method=='SMI';hy=backbone=='HY1.5';n=len(gpu_group)
 if n not in (4,8):raise ValueError('Use one group of 4 or 8 GPUs per inference job')
 recorded_n=None;topology_match=None
 workers=[f'cuda:{i}'for i in range(n)];main=workers[-1]
 infer_mask=','.join(str(i)for i in gpu_group);judge_mask=infer_mask
 qkey=HY_QWEN if hy else WAN_QWEN;qmodel=models['qwen_checkpoints'][qkey]
 common={'PYTHONPATH':os.pathsep.join([str(clone),str(clone/'wan'),str(code/'vendor/SpatialStack'),str(code/'vendor/SpatialStack/src')]),'OMP_NUM_THREADS':'3','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TORCHDYNAMO_DISABLE':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','NCCL_DEBUG':'WARN','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','HY_ROOT':str(clone),'SPATIALSTACK_ROOT':str(code/'vendor/SpatialStack')}
 # Only model/inference controls are accepted from old logs, never credentials or shell environment dumps.
 for k,v in env.items():
  if k.startswith(('QWEN_MEMORY_','ORIG_FOV_')) and not any(x in k for x in ['MODEL_PATH','OUTPUT_DIR','JUDGE_URL','TRACE_DIR','FAILED_REQUEST_DIR','SEMANTIC_MODEL_PATH']):common[k]=str(v)
 common['QWEN_MEMORY_OUTPUT_DIR']=str(job/'memory');common['QWEN_MEMORY_UPDATE_COMMIT_MODE']=env.get('QWEN_MEMORY_UPDATE_COMMIT_MODE','immediate'if hy else'deferred_remote')
 common.update(QWEN_MEMORY_JUDGE_DEVICE=main,QWEN_MEMORY_SELECT_WORKER_DEVICES=','.join(workers),QWEN_MEMORY_SELECT_WORKERS=str(len(workers)),QWEN_MEMORY_SELECT_DO_SAMPLE='false',QWEN_MEMORY_REQUEST_TIMEOUT_SEC='600',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',SMI_GEOMETRY_SEED=str(geometry_seed),SMI_REPRO_DIR=str(job/'reproducibility'))
 geometry={'seed':geometry_seed,'source':'seeded_cpu_sampler','historical_points_verified':False}
 if geometry_points is not None:
  if not hy:raise ValueError('--geometry-points currently applies to HY1.5 only')
  fixture=Path(geometry_points).expanduser().resolve()
  common['SMI_GEOMETRY_POINTS']=str(fixture)
  geometry.update(source='fixture',path=str(fixture),file_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
 generate_env=dict(common,CUDA_VISIBLE_DEVICES=infer_mask);judge_env=dict(common,CUDA_VISIBLE_DEVICES=judge_mask)
 cmd=[sys.executable,'-m','torch.distributed.run',f'--nproc_per_node={n}',f'--master_port={port+100}']
 judge=None
 if hy:
  p=copy.deepcopy(input_data)
  cmd += ['-m','hyvideo.generate']
  add(cmd,prompt=p['prompt'],image_path=p['image_path'],resolution='480p',aspect_ratio='16:9',video_length=p['num_frames'],seed=p['seed'],rewrite=False,sr=False,pose=p['pose'],output_path=job/'raw',model_path=models['hy15_backbone'],action_ckpt=models['hy15_action'],few_step=True,num_inference_steps=4,model_type='ar',width=832,height=480,offloading=True,use_vae_parallel=False,use_sageattn=False,use_fp8_gemm=False)
  cmd+=['--save_pre_sr_video','--with-ui','false']
  if smi:
   generate_env['QWEN_MEMORY_JUDGE_URL']=f'http://127.0.0.1:{port}/judge'
   judge=[sys.executable,'-m','hyvideo.qwen_judge','--model',qmodel,'--port',str(port),'--devices',','.join(workers),'--main-device',main]
  expected_raw_frames=p['num_frames'];seed=p['seed']
 else:
  p=copy.deepcopy(input_data)
  seed=int(p.get('seed',env.get('WAN_SEED',0)))
  p['seed']=seed; nav=job/'navigation_samples.jsonl'
  cmd += ['scripts/generate_wan22.py']
  add(cmd,jsonl_path=nav,output_dir=job/'videos',status_jsonl=job/'generation_status.jsonl',mode='qwen'if smi else'baseline',model_id=models['wan22_backbone'],ar_model_path=models['wan22_transformer'],ckpt_path=models['wan22_action'],height=480,width=832,num_inference_steps=int(env.get('NUM_INFERENCE_STEPS',50)),seed=seed,use_sample_seed=True,context_window_length=16,wan_memory_fov_top_k=3,skip_existing=False,qwen_memory_select_fov_top_k=7,qwen_memory_select_top_k=3,qwen_memory_local_context_chunks=3,qwen_memory_fov_fill_min_chunks=0,qwen_memory_allow_fallback=False,qwen_memory_select_filter_visual_quality=False,qwen_memory_select_policy='coverage_only'if smi else'fov_top2',qwen_memory_visual_quality_enable=smi,qwen_memory_update_enable=smi,qwen_memory_update_fov_top_k=6,qwen_memory_select_do_sample=False,qwen_memory_disable_sparse=not smi,qwen_memory_select_active_only=True,qwen_memory_strict=smi)
  cmd+=['--qwen_memory_sparse_member_from_ends=-2,-3']
  if smi:
   add(cmd,qwen_intermediate_root=job/'memory',qwen_memory_judge_url=f'http://127.0.0.1:{port}/judge',qwen_memory_judge_model_path=qmodel,qwen_memory_judge_device=main)
   judge=[sys.executable,'scripts/qwen_judge_wan22.py','--backend','spatialstack','--model-path',qmodel,'--spatialstack-root',str(code/'vendor/SpatialStack'),'--port',str(port),'--device',main,'--dtype','bfloat16','--max-new-tokens','2048','--select-workers',str(len(workers)),'--select-worker-devices',','.join(workers),'--disable-linear-fast-path']
  expected_raw_frames=expected_frames
 physical=list(dict.fromkeys(infer_mask.split(',')+(judge_mask.split(',')if smi else [])))
 generate_env['SMI_SEED']=judge_env['SMI_SEED']=str(seed)
 warnings=[]
 return {'case':case_name,'method':method,'cwd':str(clone),'output':str(job),'seed':seed,'input_files':input_fingerprints(p),'expected_raw_frames':expected_raw_frames,'recorded_inference_gpus':recorded_n,'inference_gpus':n,'inference_topology_source':'explicit_gpu_group','recorded_topology_matches':topology_match,'geometry':geometry,'reproduction_warnings':warnings,'judge_devices':workers if smi else[],'judge_main_device':main if smi else None,'required_physical_gpu_indices':physical,'generate_command':cmd,'generate_env':generate_env,'judge_command':judge,'judge_env':judge_env,'port':port,'navigation':p if not hy else None,'checkpoint_key':qkey if smi else None}
def wait_idle(indices,poll=30):
 consecutive=0
 while consecutive<2:
  if STOP.is_set():raise InterruptedError('Inference stopped')
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  available={r.split(',')[0].strip() for r in rows}
  if set(indices)-available:raise ValueError('Requested GPU indices do not exist: '+str(indices))
  busy=[r for r in rows if (v:=r.split(', '))[0]in indices and (int(v[2])>=1500 or int(v[3])>=5 or v[1]in processes)]
  consecutive=0 if busy else consecutive+1
  print(json.dumps({'waiting_for_gpus':indices,'busy':busy,'consecutive_idle':consecutive}),flush=True)
  if consecutive<2:STOP.wait(poll)

def stop_process(process):
 if process is None:return
 # Each subprocess owns a fresh session, so cleanup cannot touch other jobs.
 try:os.killpg(process.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:process.wait(timeout=30)
 except subprocess.TimeoutExpired:
  try:os.killpg(process.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  process.wait()

def source_digest(root):
 items=[]
 for name in ['hyvideo','wan','scripts','vendor','configs']:
  for path in sorted((root/name).rglob('*')):
   if path.is_file() and path.suffix in ['.py','.json','.sh'] and '__pycache__'not in path.parts and path.name!='models.local.json':
    items.append([path.relative_to(root).as_posix(),hashlib.sha256(path.read_bytes()).hexdigest()])
 return hashlib.sha256(json.dumps(items).encode()).hexdigest()
def execute(p):
 job=Path(p['output']);job.mkdir(exist_ok=True,parents=True)
 if (job/'finished.json').exists():raise FileExistsError('Use --resume or a fresh output directory: '+str(job))
 if p['navigation']:(job/'navigation_samples.jsonl').write_text(json.dumps(p['navigation'])+'\n')
 wait_idle(p['required_physical_gpu_indices']);server=None;generation=None;judge_log=None
 # Remove inherited inference controls; ordinary OS/runtime environment is preserved.
 base={k:v for k,v in os.environ.items()if not k.startswith(('QWEN_','HY15_','ORIG_FOV_','CUDA_','PYTHONPATH','SMI_'))}
 try:
  write(job/'running.json',{'pid':os.getpid(),'started_at':time.time(),'case':p['case'],'gpus':p['required_physical_gpu_indices']})
  if p['judge_command']:
   judge_log=(job/'judge.log').open('w');server=subprocess.Popen(p['judge_command'],cwd=p['cwd'],env=dict(base,**p['judge_env']),stdout=judge_log,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.monotonic()+1800;opener=build_opener(ProxyHandler({}))
   while True:
    if STOP.is_set():raise InterruptedError('Inference stopped')
    assert server.poll()is None,'Judge exited; inspect judge.log'
    try:
     with opener.open(f"http://127.0.0.1:{p['port']}/health",timeout=3)as response:
      if response.status==200:break
    except Exception:
     if time.monotonic()>deadline:raise TimeoutError('judge readiness timed out')
     STOP.wait(3)
  with(job/'generation.log').open('w')as log:
   generation=subprocess.Popen(p['generate_command'],cwd=p['cwd'],env=dict(base,**p['generate_env']),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while generation.poll()is None:
    if STOP.wait(2):raise InterruptedError('Inference stopped')
   if generation.returncode:raise subprocess.CalledProcessError(generation.returncode,p['generate_command'])
  videos=list(job.rglob('*.mp4'))
  if not videos:raise RuntimeError('Generator exited without an MP4')
  media=[]
  for video in videos:
   probe=subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(video)],text=True)
   streams=json.loads(probe)['streams']
   if not streams or int(streams[0].get('nb_read_frames',0))<1:raise RuntimeError('Undecodable video: '+str(video))
   media.append({'path':str(video.relative_to(job)),'stream':streams[0]})
  raw_media=[m for m in media if m['path'].startswith('raw/') or '/raw' in m['path']]
  raw_media=raw_media or media
  if p['expected_raw_frames'] is None:raise RuntimeError('No expected frame count in recorded case inputs')
  if len(raw_media)!=1 or int(raw_media[0]['stream']['nb_read_frames'])!=int(p['expected_raw_frames']):raise RuntimeError('Generated raw frame count differs from recorded case inputs')
  if any(m['stream']['width']!=832 or m['stream']['height']!=480 for m in media):raise RuntimeError('Unexpected generated resolution')
 except Exception as exc:
  write(job/'failed.json',{'completed':False,'error':str(exc),'at':time.time()})
  raise
 finally:
  stop_process(generation);stop_process(server)
  if judge_log:judge_log.close()
  (job/'running.json').unlink(missing_ok=True)
 write(job/'finished.json',{'completed':True,'finished_at':time.time(),'media':media,'source_sha256':p['source_sha256'],'pixel_exact_reproduction':False,'note':'Inference and full frame decode completed. Historical equality and repeat-run equality require separate comparisons.'})
