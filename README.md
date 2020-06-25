### Introduction
PyTorch model code used in the 1st place winner for the  [2020 Jigsaw Multilingual Kaggle competition](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification). 
Our solution is detailed in this [Kaggle forum post](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160862).



| Model | Comment |
| ----- | ------  |
|[FastText BiGRU](classifier_bigru_fasttext_tf.py) | monolingual RNN approach using non-contextualized FastText embeddings |
|[HuggingFace Transformer](classifier_baseline.py) | mono/multilingual Transformer approach |  

| Helper modules | Comment | 
| -------------- | ------- |
| [prepare_data](prepare_data.py) | Generates the prerequisite train/test/validation data necessary for training |
| [prepare_predictions](prepare_predictions.py) | Blends current run predictions with the previous ensemble |
| [preprocessor](preprocessor.py)| Includes helper functions to extract raw strings and labels from training CSVs |
| [postprocessor](postprocessor.py)| Includes helper functions to ensemble multiple CSV predictions |

### Data and model files
1. HuggingFace models are downloaded directly via API so there is no need to manually download them.
2. FastText monolingual models can be downloaded [here](https://fasttext.cc/docs/en/crawl-vectors.html). Our model code uses bin model files (e.g., cc.es.300.bin).
3. Translations of the Toxic 2018 dataset and pseudo-labels for public LB 9500 and public LB 9537 (used as sample inputs to training) can be found [here](https://www.kaggle.com/leecming/multilingual-toxic-comments-training-data).

### Setup
1. A [DockerFile](Dockerfile) is provided which builds against an Nvidia-provided Docker image, and installs the necessary Nvidia, system, and Python prerequisites - IMPORTANT: the Dockerfile installs an SSH server with a default password
2. Various functions ingest and generate files - it is suggested that you mount them within container volumes to allow for smooth movement of files in & out of the container
3. With an SSH server and Jupyter notebook server within the container - it is suggested that you bind ports to enable external connections

### Example 1: Running a spanish monolingual Transformer model using public LB 9500 pseudo-labels 
1. We use the pretrained spanish model mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es
2. Update [SETTINGS.json](SETTINGS.json) so that PSEUDO_LABELS_PATH and other paths are updated for your setup.
3. Run prepare_data.py
4. Run classifier_baseline.py
5. Run prepare_predictions.py

A submission file should be generated at $TRAIN_DATA_DIR/curr_run_submission.csv. For this, we were able to generate a 9502 public LB submission. Due to training variability, you may get different results.


### Example 2: Running a spanish monolingual FastText model using public LB9537 pseudo-labels
1. We use the pretrained official FastText embeddings for spanish (cc.es.300.bin) 
2. Update [SETTINGS.json](SETTINGS.json) so that PSEUDO_LABELS_PATH and other paths are updated for your setup.
3. Run prepare_data.py
4. Run classifier_bigru_fasttext_tf.py
5. Update prepare_predictions.py so that ENSEMBLE_WEIGHT=0.8 (give less weight to predictions from this model)
6. Run prepare_predictions.py

A submission file should be generated at $TRAIN_DATA_DIR/curr_run_submission.csv. For this, we were able to generate a 9540 public LB submission. Due to training variability, you may get different results.

### Notes
1. The code and container environment were run on a fairly heavy-weight workstation (24C/48T Threadripper + 64GB RAM + 2x RTX Titans w/ 24GB GPU RAM each) and using mixed precision training + gradient accumulation. It may not be feasible to finetune larger models such as XLM-Roberta-Large on smaller-scale machines especially with entry-level Nvidia cards.
You can adjust the BATCH_SIZE and ACCUM_FOR flags in classifier_baseline.py to fit in memory but it may impact model performance. 
