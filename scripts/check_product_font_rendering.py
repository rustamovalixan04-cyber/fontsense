"""Stress-check selected font rendering before bulk generation; no model inference."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

import pandas as pd

from fontsense.utils import load_json, project_root, save_json
from fontsense.v3_dataset import render_product_image


def main():
    root=project_root()
    config=load_json(root/'config/v3/dataset.json')
    split=pd.read_csv(root/config['split_path'])
    seeds=(42,1234,6789,131337,14629533,250099,823341,1000000049)
    def check(row):
        for seed in seeds:
            try:
                render_product_image(str(root/row['path']),seed,config)
            except Exception as exc:
                return dict(family=row['family'],seed=seed,error=f'{type(exc).__name__}: {exc}')
        return None
    started=time.perf_counter()
    failures=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(check,row) for row in split.to_dict('records')]
        for index,job in enumerate(as_completed(futures),1):
            failure=job.result()
            if failure:
                failures.append(failure)
            if index%200==0:
                print(f'Render stress check: {index}/{len(split)} families, {len(failures)} failures',flush=True)
    summary=dict(status='failed' if failures else 'passed',families_checked=len(split),seeds_per_family=list(seeds),renders_attempted=len(split)*len(seeds),failures=failures,elapsed_seconds=time.perf_counter()-started,model_images_evaluated=0,scope='Technical font-rendering checks, including reserved-family fonts; no training, prediction, threshold selection, or test evaluation')
    save_json(summary,root/'reports/v3/font_render_stress_check.json')
    print(json.dumps(summary,indent=2))
    if failures:
        raise RuntimeError('Selected fonts failed render stress checks; do not start training')


if __name__=='__main__':
    main()
