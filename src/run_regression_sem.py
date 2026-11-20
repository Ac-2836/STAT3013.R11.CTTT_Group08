"""Execute both notebooks with fresh kernels and record input/output provenance.

Run from any directory: python src/run_regression_sem.py
Use --cwd notebooks --verify-repro for a second run from the notebook directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONHASHSEED'] = '42'
os.environ['IPYTHONDIR'] = str(Path(__file__).resolve().parents[1]/'.cache'/'ipython')
os.environ['MPLCONFIGDIR'] = str(Path(__file__).resolve().parents[1]/'.cache'/'matplotlib')
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient
from regression_sem_utils import ROOT, TABLES, environment_manifest, write_json


def hashes(paths):
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwd', choices=['root','notebooks'], default='root')
    parser.add_argument('--verify-repro', action='store_true')
    args = parser.parse_args()
    # Ensure subprocess kernel entry points resolve to the interpreter running this command.
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    before = environment_manifest()
    start = time.monotonic()
    report_path = TABLES/'regression_sem_execution.json'
    previous = json.loads(report_path.read_text(encoding='utf-8')) if args.verify_repro else None
    execution = []
    for name in ['10_regression.ipynb','50_sem.ipynb']:
        path = ROOT/'notebooks'/name
        book = nbformat.read(path, as_version=4)
        for cell in book.cells:
            if cell.cell_type == 'code':
                compile(cell.source, name, 'exec')
                cell.outputs = []; cell.execution_count = None
        cwd = ROOT if args.cwd=='root' else ROOT/'notebooks'
        client = NotebookClient(book, timeout=1800, kernel_name='python3', resources={'metadata':{'path':str(cwd)}})
        print('Executing '+name+' from '+args.cwd, flush=True)
        client.execute()
        errors = [o for c in book.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
        assert not errors
        counts = [c.execution_count for c in book.cells if c.cell_type=='code']
        assert counts == list(range(1,len(counts)+1))
        nbformat.write(book, path)
        execution.append(dict(notebook=name, code_cells=len(counts), error_outputs=len(errors)))
        print('Completed '+name,flush=True)
    assert before['inputs'] == environment_manifest()['inputs'], 'Processed input changed'
    artifacts = sorted(TABLES.glob('regression_*.csv'))+sorted(TABLES.glob('sem_*.csv'))
    artifacts += [ROOT/'reports'/'predictions'/'regression.csv', TABLES/'regression_selected.json', TABLES/'sem_bootstrap_summary.json']
    outputs = hashes(artifacts)
    if previous:
        assert outputs == previous['artifact_sha256'], 'Reproduction changed a numerical artifact'
    report = dict(environment=before, cwd=args.cwd, notebooks=execution, processed_inputs_unchanged=True,
        artifact_sha256=outputs, exact_numerical_reproduction=bool(previous), seconds=round(time.monotonic()-start,2))
    write_json(report_path, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['environment','artifact_sha256']},indent=2),flush=True)


if __name__ == '__main__':
    main()
