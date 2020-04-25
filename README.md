### Introduction
A suite of primarily PyTorch modules targeting the [2020 Jigsaw Multilingual Kaggle competition](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification). The [baseline classifier](classifier_baseline.py) is self-contained and finetunes a HuggingFace pretrained model against the competition train, validation, and test sets.
Local validation is performed against the validation dataset. 

Alternative approaches (and modifications) to the baseline classifier include:
(note that these modules aren't as tested as the baseline classifier and may require debugging)

| Model | Comment |
| ----- | ------  |
|[FastText BiGRU](classifier_bigru_fasttext_tf.py) | RNN approach using non-contextualized FastText embeddings |
|[ELMO BiGRU](classifier_elmo_tf.py) | RNN approach using contextualized ELMO features| 
|[Universal Sentence Encoder](classifier_combo_use.py) | Combines a Transformer with a USE sentence encoder |
|[Transformer using histogram loss](classifier_hist_loss.py) | Converts binary labels into bins i.e., histogram loss | 
|[K-folds training](classifier_kfolds.py)| Modifies baseline to perform k-folds CV |
|[LASER classifier](classifier_mixup.py) | MLP classifier on top of Facebook LASER embeddings |
|[Mixup training](classifier_mixup.py) | Baseline with mixup applied at the embedding level | 
|[Translation pairs](classifier_pairs.py) | Baseline modified to train against raw/translation comment pairs |
|[Unsupervised data augmentation](classifier_uda.py) | Added unsupervised consistency loss applied to the test set |
|[MLM pretraining](pretraining_mlm.py) | Not a classifier but an intermediate trainer that finetunes a pretrained model against the competition dataset using MLM|

| Helper modules | Comment | 
| -------------- | ------- |
| [preprocessor](preprocessor.py)|Includes helper functions to extract raw strings and labels from training CSVs|
| [postprocessor](postprocessor.py)|Includes helper functions to ensemble multiple CSV predictions|

### Setup
1. A [DockerFile](Dockerfile) is provided which builds against an Nvidia-provided Docker image, and installs the necessary Nvidia, system, and Python prerequisites - IMPORTANT: the Dockerfile installs an SSH server with a default password
2. Various functions assume the presence of data/ and models/ subdirectories - it is suggested that you mount them as container volumes to allow for smooth movement of files in & out of the container
3. With an SSH server and Jupyter notebook server within the container - it is suggested that you bind ports to enable external connections


### Notes
1. The code and container environment were run on a fairly heavy-weight workstation (24C/48T Threadripper + 64GB RAM + 2x RTX Titans w/ 24GB GPU RAM each) and using mixed precision training + gradient accumumlation. It may not be feasible to finetune larger models such as XLM-Roberta-Large on smaller-scale machines especially with entry-level Nvidia cards. 
