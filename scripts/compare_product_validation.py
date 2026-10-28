"""Compare existing candidates on product validation only; never open test shards."""
from pathlib import Path
import argparse
import json
import time

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
from torch.utils.data import DataLoader

from fontsense.cnn_model import FontSenseCNN
from fontsense.utils import load_json, project_root, save_json
from fontsense.v2_data import sha256_file
from fontsense.v3_dataset import CLASSES
from fontsense.v3_train import ProductShardDataset, run_epoch, dataset_content_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    root=project_root()
    quality=load_json(root/'reports/v3/dataset_validation_summary.json')
    if quality.get('status')!='passed' or quality.get('images_total')!=500000:
        raise ValueError('Complete product data validation is required')
    manifest=root/'data/v3/manifests/manifest.sqlite'
    if sha256_file(manifest)!=quality['manifest_sha256']:
        raise ValueError('Manifest changed after validation')
    config=load_json(root/'config/v3/training.json')
    inventory=pd.read_csv(root/'reports/v3/shard_inventory.csv')
    content_identity=dataset_content_identity(quality,inventory)
    source=root/config['initial_checkpoint']
    if sha256_file(source)!=config['initial_checkpoint_sha256']:
        raise ValueError('Candidate B checkpoint changed')
    report_root=root/'reports/v3/validation_comparison'
    report_root.mkdir(parents=True,exist_ok=True)
    output=report_root/'candidate_b.json'
    if output.is_file():
        baseline=load_json(output)
        if baseline['dataset_content_sha256']!=content_identity or baseline['checkpoint_sha256']!=sha256_file(source):
            raise ValueError('Existing baseline comparison uses a different contract')
    else:
        device=torch.device(args.device)
        if device.type=='cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
            torch.cuda.set_per_process_memory_fraction(.7)
        torch.set_num_threads(2)
        saved=torch.load(source,map_location='cpu',weights_only=True)
        if saved['classes']!=list(CLASSES) or saved['training_data']['test_images_loaded']!=0:
            raise ValueError('Candidate B violates the class or historical test boundary')
        model=FontSenseCNN(5,**saved['architecture']).to(device)
        model.load_state_dict(saved['state_dict'])
        dataset=ProductShardDataset(root/'data/v3',inventory,'validation',tuple(saved['preprocessing']['image_size']))
        loader=DataLoader(dataset,batch_size=256,num_workers=4,pin_memory=device.type=='cuda')
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        metrics,predictions=run_epoch(model,loader,None,scaler,device,False,mixed_precision=False)
        if metrics['images']!=75000 or len(predictions)!=75000 or predictions['sample_id'].nunique()!=75000:
            raise ValueError('Baseline comparison did not cover exactly 75000 validation images')
        if not predictions['category'].value_counts().eq(15000).all():
            raise ValueError('Unexpected validation class counts')
        prediction_path=root/'data/v3/manifests/candidate_b_validation_predictions.csv'
        predictions.to_csv(prediction_path,index=False)
        baseline=dict(name='Candidate B on V3 validation',**metrics,checkpoint=config['initial_checkpoint'],checkpoint_sha256=sha256_file(source),preprocessing=saved['preprocessing'],manifest_sha256=quality['manifest_sha256'],dataset_content_sha256=content_identity,predictions_sha256=sha256_file(prediction_path),device=str(device),mixed_precision=False,end_to_end_milliseconds_per_image=metrics['seconds']*1000/metrics['images'],test_images_loaded=0,real_world_accuracy_claim=False)
        save_json(baseline,output)
        save_json(classification_report(predictions['category'],predictions['predicted_category'],labels=list(CLASSES),output_dict=True,zero_division=0),report_root/'candidate_b_classification_report.json')
        majority=['display']*len(predictions)
        save_json(dict(name='Majority-class sanity check (balanced train; deterministic display tie-break)',accuracy=accuracy_score(predictions['category'],majority),macro_f1=f1_score(predictions['category'],majority,average='macro',zero_division=0),validation_images=len(predictions),fit_source='Balanced training class counts only',test_images_loaded=0),report_root/'majority.json')
    comparison=[dict(model='Majority sanity check',**{key:load_json(report_root/'majority.json')[key] for key in ('accuracy','macro_f1')}),dict(model='V2 Candidate B',accuracy=baseline['accuracy'],macro_f1=baseline['macro_f1'])]
    product_summary=root/'reports/v3/training/summary.json'
    if product_summary.is_file():
        product=load_json(product_summary)
        if product['dataset_content_sha256']!=content_identity:
            raise ValueError('Product candidate uses a different validation contract')
        comparison.append(dict(model='V3 product candidate',accuracy=product['validation_accuracy'],macro_f1=product['validation_macro_f1']))
    pd.DataFrame(comparison).to_csv(report_root/'comparison.csv',index=False)
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    main()
