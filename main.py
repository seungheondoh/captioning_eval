import os
import random
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
# ms coco evalu
from eval_metrics import evaluate_metrics
# hf eval
import evaluate

def inference_parsing(inference_results):
    # inference parsing
    ids, predictions, ground_truths = [], [], []
    for k, v in inference_results.items():
        ids.append(k)
        predictions.append({
            "file_name": k,
            "caption_predicted":v['predictions']
        })
        ground_truths.append({
            "file_name": k,
            'caption_reference_{:02d}'.format(1):v['true_captions']
        })
    return predictions, ground_truths

def main(args):
    inference_results = json.load(open("./samples/inference_results.json", 'r'))
    
    if args.types == "coco_eval":
        predictions, ground_truths = inference_parsing(inference_results)
        coco_metric = evaluate_metrics(predictions, ground_truths, 1)
        for k,v in coco_metric.items():
            print(k, v['score'])

    elif args.types == "hf_eval":
        bleu = evaluate.load("bleu")
        rouge = evaluate.load('rouge')
        meteor = evaluate.load('meteor')
        bertscore = evaluate.load("bertscore")

        predictions, references = [], []
        for k,v in tqdm(inference_results.items()):
            predictions.append(v['predictions'])
            references.append(v['true_captions'])

        bleu1_results = bleu.compute(predictions=predictions, references=references, max_order=1)
        bleu2_results = bleu.compute(predictions=predictions, references=references, max_order=2)
        bleu3_results = bleu.compute(predictions=predictions, references=references, max_order=3)
        bleu4_results = bleu.compute(predictions=predictions, references=references, max_order=4)
        meteor_results = meteor.compute(predictions=predictions, references=references)
        rouge_results = rouge.compute(predictions=predictions, references=references)
        bertscore_results = bertscore.compute(predictions=predictions, references=references, lang='en')
        print({
            "bleu1":bleu1_results['bleu'], 
            "bleu2":bleu2_results['bleu'], 
            "bleu3":bleu3_results['bleu'], 
            "bleu4":bleu4_results['bleu'], 
            "meteor": meteor_results['meteor'],
            "rougeL": rouge_results['rougeL'],
            "bertscore": np.mean(bertscore_results['f1'])
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", default="coco_eval", choices=['coco_eval', 'hf_eval'], type=str)
    args = parser.parse_args()
    main(args=args)
