"""Build reproducible standalone and Kaggle Notebook artifacts.

The agent is always packaged as main.py at the archive root (Kaggle requires
that name), but it can be built from any source file:

    python build_submission.py                 # packages main.py
    python build_submission.py --agent main_v3.py
"""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
import nbformat

parser=argparse.ArgumentParser()
parser.add_argument('--agent',default='agent/main.py',
                    help='source file to package as main.py in the archive')
args=parser.parse_args()

root=Path(__file__).resolve().parents[1]
source=(root/args.agent).read_text()
print(f'packaging {args.agent} as main.py')
with tarfile.open(root/'submission.tar.gz','w:gz') as archive:
    payload=source.encode(); info=tarfile.TarInfo('main.py'); info.size=len(payload); info.mtime=0
    archive.addfile(info,io.BytesIO(payload))
nb=nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell('# Kaggriculture — coordinated market-aware farming\n\nStandalone Python agent. Uses crop profitability forecasts, coordinated workers, conservative seed purchasing and explicit final-day liquidation. Local results are not a leaderboard ranking. Run all cells, save a notebook version, then submit `submission.tar.gz` to Kaggriculture.'),
    nbformat.v4.new_code_cell('from pathlib import Path\n\nAGENT_SOURCE = '+repr(source)+'\nPath("main.py").write_text(AGENT_SOURCE)\nprint("Created main.py")'),
    nbformat.v4.new_code_cell('from kaggle_environments import make\n\nenv = make("kaggriculture", configuration={"seed": 601}, debug=True)\nenv.run(["main.py", "main.py"])\nstatuses = [s.status for s in env.state]\nassert statuses == ["DONE", "DONE"], statuses\nprint("Self-play validation:", [(s.status, s.reward) for s in env.state])'),
    nbformat.v4.new_code_cell('import tarfile\n\nwith tarfile.open("submission.tar.gz", "w:gz") as archive:\n    archive.add("main.py", arcname="main.py")\nprint("Ready: submission.tar.gz")'),
])
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nbformat.validate(nb)
nbformat.write(nb,root/'kaggriculture_agent.ipynb')
manifest={name:{'bytes':(root/name).stat().st_size,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in [args.agent,'submission.tar.gz','kaggriculture_agent.ipynb']}
manifest['packaged_from']=args.agent
(root/'results/artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
