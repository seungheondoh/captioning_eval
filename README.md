# Music captioning evaluation metrics

This repository contains code to evaluate translation metrics on music captioning predictions.
We compare [MScoco evaluate](https://github.com/tylin/coco-caption) and [huggingface evaluate](https://github.com/huggingface/evaluate).

## Quick Start

```python
cd coco_caption
bash get_stanford_models.sh
pip install -r requirements.txt
python main.py --types coco_eval # hf_eval
```

## Results

|           | COCO_eval | HF_eval | Abs Diff |
|-----------|-----------|---------|----------|
| bleu1     | 0.28      | 0.29    | 0.01     |
| bleu2     | 0.14      | 0.15    | 0.01     |
| bleu3     | 0.08      | 0.09    | 0.01     |
| bleu4     | 0.05      | 0.06    | 0.01     |
| METEOR    | 0.11      | 0.21    | 0.10     |
| RougeL    | 0.22      | 0.22    | 0.00     |
| CIDEr     | 0.07      | -       | -        |
| SPICE     | 0.09      | -       | -        |
| BertScore | -         | 0.86    | -        |
| SenBERT   | -         | -       | -        |

## Data I/O
We proceed with the evaluation using the music caps dataset (eval set). You can check the `prediction` and `groundturth` data in `samples/inference_results.json`. The baseline system is the muscaps model (manco et al.).

```
    "19": {
        "predictions": " This song is an instrumental. The song is medium tempo with a steady drumming rhythm, groovy bass line and keyboard accompaniment. The song is exciting and energetic. The song is a modern pop song with poor audio quality. ",
        "true_captions": "You can hear two people playing various percussive instruments. One is holding the same beat playing on congas while the other is playing a solo changing rhythms and percussive sounds in a complex manner. This song may be playing live demonstrating a solo run.",
        "audio_paths": "/muscaps/data/datasets/audiocaption/audio/[-R0267o4lLk]-[60-70].npy"
    },
    "20":{
        ..
    },
```


## Coco eval inputs

The code from the Microsoft COCO caption evaluation repository, in the folder coco_caption, is used to evaluate the metrics. The code has been refactored to work with Python 3 and to also evaluate the SPIDEr metric. Image-specific names and comments in-code were also changed to be audio-specific. SPICE evaluation uses 8GB of RAM and METEOR uses 2GB (both use Java). To limit RAM usage go to coco_caption/pycocoevalcap and meteor/meteor.py:18 or spice/spice.py:63 respectively and change the third argument of the java command.

The input files can be given either as file paths (string or pathlib.Path) or lists of dicts with a dict for each row, the dicts having the column headers as keys (as given by csv.DictReader in Python). The prediction file must have the fields file_name and caption_predicted and the reference file must have the fields file_name and caption_reference_XX with XX being the two-digit index of the caption, e.g. caption_reference_01,...,caption_reference_05 with five reference captions. (we use only caption_reference_01)

The metric evaluation function outputs the evaluated metrics in a dict with the lower case metric names as keys. One score is evaluated for each audio file and its predicted caption. Additionally, a single score is evaluated for the whole dataset. The format of the output is the following:

### Plan

- [] add bert score
- [] add sentence bert score


### Reference
- https://github.com/tylin/coco-caption.
- https://github.com/audio-captioning/caption-evaluation-tools