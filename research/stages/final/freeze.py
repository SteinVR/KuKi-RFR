"""Freeze the final seven systems using development artifacts only."""
import json
from datetime import datetime, timezone
from pathlib import Path
from research.kuki.data import digest, write_json

ROOT = Path('research/stages/final')
ST = Path('research/stages')
def read(p):
    return json.loads(Path(p).read_text())

def main():
    assert not (ROOT/'manifest.json').exists()
    paths = {
        'l1': ST/'ST5-l1-roles/search/experiments/20260929-l1-search-dropout-s42-01',
        'l2': ST/'ST3-l2-l3-joint/experiments/20260929-l2-only-s42-01',
        'l3': ST/'ST3-l2-l3-joint/experiments/20260929-l3-only-s42-01',
        'st3': ST/'ST3-l2-l3-joint/search/experiments/20260929-joint200-s42-01',
        'st6': ST/'ST6-l1-l2-l3/balance/experiments/20260929-st6-balance-roles-s42-01',
        'st4_control': ST/'ST4-l3-to-l2/frozen/experiments/20260929-st4-frozen-control-s42-01',
        'st4_families': ST/'ST4-l3-to-l2/frozen/experiments/20260929-st4-frozen-families-s42-01',
    }
    systems = {}
    for name, run in paths.items():
        metric_path = run/('best/dev-metrics.json' if name in ['l1','st6'] else 'best/metrics.json')
        metrics = read(metric_path)
        if 'dev' in metrics:
            metrics = metrics['dev']
        cfg = read(run/'config.json')
        p = read(run/'progress.json')
        systems[name] = dict(run=str(run), checkpoint=str(run/'checkpoints/best.pt'),
                            epoch=p['best_epoch'], packed=cfg, dev_metrics=metrics,
                            conditional_threshold=metrics.get('conditional',{}).get('threshold'),
                            checkpoint_sha256=digest(run/'checkpoints/best.pt'))
    detector = ST/'ST2-l3-rhetoric/experiments/20260928-detector-base-s42-01'
    files = [Path('uv.lock'),Path('research/configs/data.json'),Path('research/configs/encoder.json'),ROOT/'protocol.md']
    files += list(Path('research/kuki').glob('*.py')) + list(ROOT.glob('*.py'))
    for run in [*paths.values(), detector]:
        files += [p for p in run.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    for stage in ['ST3-l2-l3-joint','ST4-l3-to-l2/frozen','ST6-l1-l2-l3','ST2-l3-rhetoric/modular']:
        files += list((ST/stage).glob('*.py'))
    dc = read('research/configs/data.json')
    files += [Path(dc['split_manifest']),Path(dc['prepared_path'])]
    manifest = dict(frozen_at=datetime.now(timezone.utc).isoformat(), systems=systems,
                    detector=dict(run=str(detector), checkpoint_sha256=digest(detector/'checkpoints/best.pt')),
                    files={str(p):digest(p) for p in sorted(set(files))}, test_opened=False,
                    bootstrap=dict(replicates=10000,seed=42,unit='paired article',confidence=.95),
                    decoder=dict(alpha=2.,threshold=-5.172313813576109,top_candidates=500,top_outputs=100),
                    l1_decision='argmax', l2_threshold=.5)
    write_json(ROOT/'manifest.json',manifest)
    print({k:dict(epoch=v['epoch'],conditional_threshold=v['conditional_threshold']) for k,v in systems.items()})

if __name__=='__main__':
    main()
