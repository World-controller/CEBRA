import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
CORE_HASHES={
 'src/evaluation/cmrg_calibrated.py':'f2aea1e721145bd9288f3637a21b537af6dfbb5d9619d120c34cc60589ab5bf2',
 'src/evaluation/run_four_configs.py':'6f54083a203a52a2b5f19ea5f7a4102399173d6d9ebc21bbb0066e92fa395fe7',
 'src/evaluation/cmrg_controls.py':'9230b51b2093cb4adab3f893a3a2ab6511c2c8728337c63ae3867acffabdb281',
 'src/evaluation/cmrg_retrieve_metrics.py':'f6b4b780ee58f9d4dfda13f6b37a59976b2e75e862670efd97f95572d14c5aab',
}

def test_core_files_unchanged():
    for name,expected in CORE_HASHES.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected


def test_readme_branding_and_assets():
    text=(ROOT/'README.md').read_text()
    assert text.startswith('# CEBRA\n')
    assert 'EfficientRAG-official' not in text
    assert 'State-Aware Chunk Reduction for Efficient Context Construction in Multi-Hop RAG' in text
    assert 'synthetic smoke example' in text
    assert 'Built with Meta Llama 3.' in text
    assert not (ROOT/'FULL_WORKSPACE_SNAPSHOT.md').exists()
    assert '(assets/framework-1.png)' in text
    assert '(assets/framework-1.pdf)' in text
    assert 'cebra-overview.svg' not in text
    assert not (ROOT/'assets/cebra-overview.svg').exists()


def test_manuscript_framework_assets():
    original=(ROOT/'assets/framework-1.pdf').read_bytes()
    assert original.startswith(b'%PDF-')
    # Preserve the manuscript PDF byte-for-byte; the PNG is a display-only render.
    assert hashlib.sha256(original).hexdigest()=='63be630f0a06c0df7f145570d579e7fb0577cf2ec14b121e5f337a944e0ad5cc'
    preview=(ROOT/'assets/framework-1.png').read_bytes()
    assert preview.startswith(b'\x89PNG\r\n\x1a\n')
    assert int.from_bytes(preview[16:20],'big')>=3000
    assert int.from_bytes(preview[20:24],'big')>0


def test_local_markdown_links_exist():
    docs=[ROOT/'README.md',*list((ROOT/'docs').glob('*.md'))]
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if not target.startswith(('https://','http://','#','mailto:')):
                assert (path.parent/target.split('#')[0]).exists(),(path,target)


def test_required_public_files_not_ignored():
    if not (ROOT/'.git').exists():return
    names=[str(p.relative_to(ROOT)) for directory in ['assets','configs','models','examples/smoke'] for p in (ROOT/directory).iterdir() if p.is_file()]
    result=subprocess.run(['git','check-ignore',*names],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==1,result.stdout


def test_license_and_attribution_preserved():
    assert 'Copyright (c) Microsoft Corporation.' in (ROOT/'LICENSE').read_text()
    assert 'permission notice shall be included' in (ROOT/'LICENSE').read_text()
    assert 'Meta Llama 3' in (ROOT/'NOTICE').read_text()
    assert 'NIL-zhuang/EfficientRAG-official' in (ROOT/'docs/third-party.md').read_text()


def test_no_private_material_in_new_public_text():
    selected=[ROOT/'README.md',ROOT/'src/cebra.py',*list((ROOT/'docs').glob('*.md')),*list((ROOT/'configs').glob('*.json')),*list((ROOT/'models').glob('*.json'))]
    for p in selected:
        s=p.read_text()
        assert not re.search(r'/Users/|/root/|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}',s),p


def test_synthetic_split_disjointness_and_manifest():
    ids={split:[json.loads(l)['id'] for l in (ROOT/f'examples/smoke/{split}.jsonl').read_text().splitlines()] for split in ['train','calibration','test']}
    assert not set(ids['train'])&set(ids['calibration'])
    assert not set(ids['train'])&set(ids['test'])
    assert not set(ids['calibration'])&set(ids['test'])
    manifest=json.loads((ROOT/'examples/smoke/calibration.manifest.json').read_text())
    assert manifest['question_ids']==ids['calibration']
    assert manifest['sha256']==hashlib.sha256((ROOT/'examples/smoke/calibration.jsonl').read_bytes()).hexdigest()
