"""Technical GPU/data-loader preflight; it does not produce model performance results."""
from pathlib import Path
import argparse
import json
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fontsense.cnn_model import FontSenseCNN
from fontsense.utils import project_root, save_json
from fontsense.v3_train import ProductShardDataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=str,default='gpu_preflight.json')
    args=parser.parse_args()
    root=project_root()
    data=root/'data/v3/cache/pilot_render'
    inventory=pd.DataFrame([json.loads(path.read_text()) for path in (data/'manifests/receipts').glob('*.json')])
    dataset=ProductShardDataset(data,inventory,'train',(192,64),shuffle_buffer=256)
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA preflight failed')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70)
    model=FontSenseCNN(5,24,.28).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
    scaler=torch.amp.GradScaler('cuda')
    started=time.perf_counter()
    seen=0
    warmup_seconds=0.0
    warmup_images=0
    for index,(images,labels,_,_) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            loss=torch.nn.functional.cross_entropy(model(images.cuda(non_blocking=True)),labels.cuda(non_blocking=True))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss in preflight')
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        seen+=len(labels)
        if index==3:
            torch.cuda.synchronize()
            warmup_seconds=time.perf_counter()-started
            warmup_images=seen
        if index>=29:
            break
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    result=dict(status='passed',scope='Technical loader, mixed-precision, backward-pass and memory preflight only. Disposable weights; no accuracy results or released checkpoint.',gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_version=torch.version.cuda,batch_size=args.batch_size,loader_workers=args.workers,images_processed=seen,elapsed_seconds=elapsed,images_per_second=seen/elapsed,steady_images_per_second=(seen-warmup_images)/(elapsed-warmup_seconds),peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,training_input_shape=[1,64,192],test_images_loaded=0)
    save_json(result,root/'reports/v3'/args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
