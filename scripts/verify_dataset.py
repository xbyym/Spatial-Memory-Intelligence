"""Verify released annotations and image shards; optionally extract training images."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, zipfile

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'datasets')
    p.add_argument('--extract',action='store_true');args=p.parse_args();root=args.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    images={r['path']:r for r in map(json.loads,(root/'images_manifest.jsonl').read_text().splitlines())}
    seen=set()
    for shard in manifest['image_shards']:
        path=root/shard['path'];assert sha(path)==shard['sha256'],path
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                row=images[name];content=z.read(name)
                assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256'],name
                assert name not in seen;seen.add(name)
                if args.extract:
                    target=(root/name).resolve();assert target.is_relative_to(root/'images')
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    assert seen==set(images)
    for model,splits in manifest['datasets'].items():
        for split,spec in splits.items():
            f=root/spec['path'];assert sha(f)==spec['sha256'];rows=json.loads(f.read_text())
            assert len(rows)==spec['rows'] and dict(Counter(r['task'] for r in rows))==spec['task_counts']
            for row in rows:
                for image in row['images']:assert image in images,(model,split,row['id'],image)
    print(json.dumps({'verified':True,'images':len(images),'extracted':args.extract,'datasets':list(manifest['datasets'])}))

if __name__=='__main__':main()
