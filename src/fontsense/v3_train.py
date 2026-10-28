"""Stream the validated product shards and train with resumable epoch checkpoints."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
import tarfile

from .cnn_model import FontSenseCNN
from .utils import project_root, load_json, save_json, set_seed
from .v2_data import sha256_file
from .v3_dataset import CLASSES


class ProductShardDataset(IterableDataset):
    """Only pixels enter the network; metadata supplies labels and diagnostics."""
    def __init__(self, root: Path, inventory: pd.DataFrame, split: str, image_size: tuple[int,int], seed: int=42, shuffle_buffer: int=1024):
        if split not in {'train','validation'}:
            raise ValueError('Training pipeline forbids test-shard access')
        self.root,self.split,self.seed,self.epoch,self.shuffle_buffer=Path(root),split,seed,0,shuffle_buffer
        self.shards=inventory.loc[inventory['split']==split].sort_values('shard_index').to_dict('records')
        if not self.shards:
            raise ValueError(f'No {split} shards')
        width,height=image_size
        operations=[transforms.Grayscale(1),transforms.Resize((height,width),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)]
        if split=='train':
            operations.append(transforms.RandomAffine(1.0,translate=(.01,.02),interpolation=transforms.InterpolationMode.BILINEAR,fill=127))
        operations.extend([transforms.ToTensor(),transforms.Normalize((.5,),(.5,))])
        self.transform=transforms.Compose(operations)

    def __len__(self):
        return sum(int(row['images']) for row in self.shards)

    def __iter__(self):
        worker=get_worker_info()
        worker_id=worker.id if worker else 0
        worker_count=worker.num_workers if worker else 1
        seed=self.seed+self.epoch*10007+worker_id
        rng=random.Random(seed)
        torch.manual_seed(seed)
        shards=list(self.shards)
        if self.split=='train':
            random.Random(self.seed+self.epoch*10007).shuffle(shards)
        shards=shards[worker_id::worker_count]
        buffer=[]

        def decode(sample):
            payload,row=sample
            with Image.open(BytesIO(payload)) as opened:
                image=self.transform(opened.convert('RGB'))
            return image,CLASSES.index(row['category']),row['sample_id'],row['family']

        for shard in shards:
            if 'archive_sha256' in shard and 'metadata_sha256' in shard:
                if sha256_file(self.root/shard['archive'])!=shard['archive_sha256'] or sha256_file(self.root/shard['metadata'])!=shard['metadata_sha256']:
                    raise ValueError('Shard or metadata changed after validation; no images from this shard were decoded')
            with gzip.open(self.root/shard['metadata'],'rt',encoding='utf-8',newline='') as handle:
                metadata={row['member']:row for row in csv.DictReader(handle)}
            with tarfile.open(self.root/shard['archive'],'r|') as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    row=metadata.get(member.name)
                    if row is None or row['split']!=self.split or row['category'] not in CLASSES:
                        raise ValueError('Shard labels or split differ from the manifest')
                    extracted=archive.extractfile(member)
                    if extracted is None:
                        raise ValueError('Missing image member')
                    sample=(extracted.read(),row)
                    if self.split=='train':
                        buffer.append(sample)
                        if len(buffer)>=self.shuffle_buffer:
                            index=rng.randrange(len(buffer))
                            yield decode(buffer.pop(index))
                    else:
                        yield decode(sample)
        while buffer:
            yield decode(buffer.pop(rng.randrange(len(buffer))))


def _atomic_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def _update_patience(score: float, meaningful_best: float, stale: int, minimum_improvement: float) -> tuple[float,int]:
    """Early stopping uses a minimum delta; checkpointing still keeps every new best."""
    if score>meaningful_best+minimum_improvement:
        return score,0
    return meaningful_best,stale+1


def dataset_content_identity(quality: dict, inventory: pd.DataFrame) -> str:
    """Identify immutable samples, not SQLite's changeable page-layout bytes."""
    columns=['split','shard_index','images','archive','archive_sha256','metadata','metadata_sha256']
    if not set(columns).issubset(inventory.columns):
        raise ValueError('The product shard inventory must include archive and metadata hashes')
    rows=inventory[columns].sort_values(['split','shard_index']).to_dict('records')
    content=dict(config_sha256=quality['config_sha256'],split_sha256=quality['split_sha256'],shards=rows)
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run_epoch(model, loader, optimizer, scaler, device, training: bool, mixed_precision: bool=True) -> tuple[dict,pd.DataFrame]:
    model.train(training)
    criterion=nn.CrossEntropyLoss()
    actual,predicted,records=[],[],[]
    total_loss,seen=0.0,0
    started=time.perf_counter()
    for images,labels,keys,families in loader:
        images,labels=images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and mixed_precision):
            logits=model(images)
            loss=criterion(logits,labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training/validation loss; last complete checkpoint is preserved')
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(optimizer)
            scaler.update()
        probabilities=torch.softmax(logits.detach().float(),dim=1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError('Non-finite model probabilities')
        targets=labels.detach().cpu().numpy()
        predictions=probabilities.argmax(axis=1)
        actual.extend(targets.tolist());predicted.extend(predictions.tolist())
        count=len(targets);seen+=count;total_loss+=float(loss.detach())*count
        if not training:
            for index,key in enumerate(keys):
                records.append(dict(sample_id=key,family=families[index],category=CLASSES[int(targets[index])],predicted_category=CLASSES[int(predictions[index])],**{f'probability_{category}':float(probabilities[index,class_index]) for class_index,category in enumerate(CLASSES)}))
        if training and seen%16384==0:
            print(f'Training images this epoch: {seen:,}',flush=True)
    if not seen:
        raise ValueError('Dataset produced no images')
    return dict(loss=total_loss/seen,accuracy=float(accuracy_score(actual,predicted)),macro_f1=float(f1_score(actual,predicted,average='macro',zero_division=0)),images=seen,seconds=time.perf_counter()-started),pd.DataFrame(records)


def train(config_path: Path, data_root: Path, report_root: Path, artifact_root: Path, device_name: str='cuda') -> dict:
    import mlflow
    root=project_root()
    config_path,data_root,report_root,artifact_root=[Path(path).resolve() for path in (config_path,data_root,report_root,artifact_root)]
    config=load_json(config_path)
    quality=load_json(root/'reports/v3/dataset_validation_summary.json')
    if quality.get('status')!='passed' or quality.get('images_total')!=500000:
        raise ValueError('The complete 500,000-image dataset must pass validation before product training')
    manifest=data_root/'manifests/manifest.sqlite'
    if sha256_file(manifest)!=quality['manifest_sha256']:
        raise ValueError('Product manifest changed after data validation')
    if sha256_file(root/config['split_path'])!=quality['split_sha256']:
        raise ValueError('Frozen product split changed after validation')
    independent=load_json(root/'reports/v3/independent_image_verification.json')
    if independent.get('status')!='passed' or independent.get('images_reopened')!=500000 or independent.get('manifest_sha256')!=quality['manifest_sha256']:
        raise ValueError('The current dataset must pass independent image verification before training')
    inventory=pd.read_csv(root/'reports/v3/shard_inventory.csv')
    content_identity=dataset_content_identity(quality,inventory)
    report_root.mkdir(parents=True,exist_ok=True)
    artifact_root.mkdir(parents=True,exist_ok=True)
    state_path=root/'artifacts/v3/training_state/latest.pt'
    best_path=artifact_root/'product_candidate.pt'
    signature=hashlib.sha256((sha256_file(config_path)+content_identity+sha256_file(__file__)).encode()).hexdigest()
    summary_path=report_root/'summary.json'
    if summary_path.is_file():
        previous=load_json(summary_path)
        if previous.get('contract_signature')==signature and best_path.is_file() and sha256_file(best_path)==previous.get('checkpoint_sha256') and previous.get('status')=='completed':
            return {**previous,'status':'reused'}
    device=torch.device(device_name)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('The isolated GPU environment cannot access CUDA')
    set_seed(config['seed'])
    torch.set_num_threads(2)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    model=FontSenseCNN(5,width=config['width'],dropout=config['dropout']).to(device)
    for module in model.modules():
        if isinstance(module,nn.BatchNorm2d):
            module.momentum=config['batch_norm_momentum']
    source=root/config['initial_checkpoint']
    if sha256_file(source)!=config['initial_checkpoint_sha256']:
        raise ValueError('Initialization checkpoint hash mismatch')
    original=torch.load(source,map_location='cpu',weights_only=True)
    if original['classes']!=list(CLASSES) or original['training_data']['test_images_loaded']!=0:
        raise ValueError('Initialization violates the class or test boundary')
    model.load_state_dict(original['state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['max_epochs'],eta_min=config['minimum_learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and config['mixed_precision'])
    history=[];best=-1.0;meaningful_best=-1.0;best_epoch=0;stale=0;start_epoch=1;training_seconds=0.0;run_id=None
    if state_path.is_file():
        state=torch.load(state_path,map_location=device,weights_only=False)
        if state['contract_signature']!=signature:
            raise ValueError('Resume contract changed; preserve the previous run and use a new run configuration')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        history,best,best_epoch,stale,start_epoch,training_seconds,run_id=state['history'],state['best'],state['best_epoch'],state['stale'],state['epoch']+1,state['training_seconds'],state['run_id']
        meaningful_best=state.get('meaningful_best',best)
        random.setstate(state['random_state']);np.random.set_state(state['numpy_state']);torch.set_rng_state(state['torch_state'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_states']])
    train_dataset=ProductShardDataset(data_root,inventory,'train',tuple(config['image_size']),config['seed'],config['shuffle_buffer'])
    validation_dataset=ProductShardDataset(data_root,inventory,'validation',tuple(config['image_size']),config['seed'])
    loader_options=dict(batch_size=config['batch_size'],num_workers=config['loader_workers'],pin_memory=device.type=='cuda')
    train_loader=DataLoader(train_dataset,**loader_options)
    validation_loader=DataLoader(validation_dataset,**loader_options)
    mlflow.set_tracking_uri('sqlite:///'+(root/'reports/v3/mlruns/mlflow.db').as_posix())
    (root/'reports/v3/mlruns').mkdir(parents=True,exist_ok=True)
    if mlflow.get_experiment_by_name(config['experiment_name']) is None:
        mlflow.create_experiment(config['experiment_name'],artifact_location=(root/'reports/v3/mlruns/artifacts').as_uri())
    mlflow.set_experiment(config['experiment_name'])
    stopped=False
    with mlflow.start_run(run_id=run_id,run_name=config['run_name']) as active:
        run_id=active.info.run_id
        mlflow.log_params({key:json.dumps(value) if isinstance(value,(list,dict)) else value for key,value in config.items()})
        mlflow.log_params(dict(dataset_images=500000,train_images=350000,validation_images=75000,test_images_loaded=0,dataset_content_sha256=content_identity,device=str(device),torch_version=torch.__version__,cuda_version=str(torch.version.cuda),source_sha256=sha256_file(__file__)))
        for epoch in range(start_epoch,config['max_epochs']+1):
            if (report_root/'STOP').exists() or training_seconds>=config['max_training_seconds']:
                stopped=True;break
            train_dataset.epoch=epoch
            training, _=run_epoch(model,train_loader,optimizer,scaler,device,True,config['mixed_precision'])
            validation,predictions=run_epoch(model,validation_loader,None,scaler,device,False,config['validation_mixed_precision'])
            if training['images']!=350000 or validation['images']!=75000:
                raise ValueError('An epoch did not cover the complete train and validation splits')
            training_seconds+=training['seconds']+validation['seconds']
            row=dict(epoch=epoch,train_loss=training['loss'],train_accuracy=training['accuracy'],train_macro_f1=training['macro_f1'],validation_loss=validation['loss'],validation_accuracy=validation['accuracy'],validation_macro_f1=validation['macro_f1'],learning_rate=optimizer.param_groups[0]['lr'],epoch_seconds=training['seconds']+validation['seconds'])
            history.append(row)
            pd.DataFrame(history).to_csv(report_root/'history.csv',index=False)
            mlflow.log_metrics({key:value for key,value in row.items() if key!='epoch'},step=epoch)
            if validation['macro_f1']>best:
                best,best_epoch=validation['macro_f1'],epoch
                payload=dict(state_dict={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},classes=list(CLASSES),architecture=dict(width=config['width'],dropout=config['dropout']),preprocessing=dict(image_size=config['image_size'],grayscale=True,normalize_mean=[.5],normalize_std=[.5]),selected_validation_run=dict(name=config['run_name'],best_epoch=best_epoch,validation_macro_f1=best,validation_accuracy=validation['accuracy']),training_data=dict(train_rows=350000,validation_rows=75000,fit_splits=['train'],selection_split='validation',test_images_loaded=0,test_rows_evaluated=0,test_metrics_recorded=False),seed=config['seed'],contract_signature=signature)
                _atomic_save(payload,best_path)
                predictions.to_csv(data_root/'manifests/best_validation_predictions.csv',index=False)
                report=classification_report(predictions['category'],predictions['predicted_category'],labels=list(CLASSES),output_dict=True,zero_division=0)
                save_json(report,report_root/'classification_report.json')
                matrix=confusion_matrix(predictions['category'],predictions['predicted_category'],labels=list(CLASSES))
                pd.DataFrame(matrix,index=CLASSES,columns=CLASSES).to_csv(report_root/'confusion_matrix.csv')
                mlflow.log_metrics(dict(best_validation_macro_f1=best,best_validation_accuracy=validation['accuracy']),step=epoch)
            meaningful_best,stale=_update_patience(validation['macro_f1'],meaningful_best,stale,config['minimum_improvement'])
            scheduler.step()
            _atomic_save(dict(contract_signature=signature,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),history=history,best=best,meaningful_best=meaningful_best,best_epoch=best_epoch,stale=stale,epoch=epoch,training_seconds=training_seconds,run_id=run_id,random_state=random.getstate(),numpy_state=np.random.get_state(),torch_state=torch.get_rng_state(),cuda_states=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),state_path)
            save_json(dict(status='training',epochs_completed=epoch,best_epoch=best_epoch,best_validation_macro_f1=best,training_seconds=training_seconds,checkpoint=str(best_path.relative_to(root)),test_images_loaded=0),report_root/'progress.json')
            print(f'V3 epoch {epoch}: train_f1={training["macro_f1"]:.4f} validation_f1={validation["macro_f1"]:.4f} best={best:.4f} seconds={row["epoch_seconds"]:.1f}',flush=True)
            if stale>=config['early_stopping_patience']:
                break
            if training_seconds>=config['max_training_seconds']:
                stopped=True;break
        if not best_path.is_file():
            raise RuntimeError('No completed training epoch/checkpoint is available')
        saved=torch.load(best_path,map_location='cpu',weights_only=True)
        summary=dict(status='intentionally_stopped' if stopped else 'completed',contract_signature=signature,best_epoch=best_epoch,epochs_completed=len(history),validation_macro_f1=best,validation_accuracy=saved['selected_validation_run']['validation_accuracy'],training_seconds=training_seconds,checkpoint=best_path.relative_to(root).as_posix(),checkpoint_sha256=sha256_file(best_path),model_size_bytes=best_path.stat().st_size,parameter_count=sum(parameter.numel() for parameter in model.parameters()),class_order=list(CLASSES),preprocessing=saved['preprocessing'],manifest_sha256=quality['manifest_sha256'],split_sha256=quality['split_sha256'],train_images_fitted=350000,validation_images_compared=75000,test_images_loaded=0,test_rows_evaluated=0,mlflow_run_id=run_id,device=str(device))
        summary.update(dataset_content_sha256=content_identity,mixed_precision=config['mixed_precision'] and device.type=='cuda',validation_mixed_precision=config['validation_mixed_precision'] and device.type=='cuda',torch_version=str(torch.__version__),gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,peak_gpu_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else 0)
        save_json(summary,summary_path)
        mlflow.log_artifact(str(best_path));mlflow.log_artifact(str(summary_path));mlflow.log_artifact(str(config_path))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=project_root()/'config/v3/training.json')
    parser.add_argument('--data',type=Path,default=project_root()/'data/v3')
    parser.add_argument('--reports',type=Path,default=project_root()/'reports/v3/training')
    parser.add_argument('--artifacts',type=Path,default=project_root()/'artifacts/v3/cnn')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    print(json.dumps(train(args.config,args.data,args.reports,args.artifacts,args.device),indent=2))
