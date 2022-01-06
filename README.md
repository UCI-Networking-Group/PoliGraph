# Privacy Policy Analyzer

The NLP-based privacy policy analyzer.

## Dependencies

The conda's internal package manager takes forever to resolve dependencies. I recommend you use [mamba](https://github.com/mamba-org/mamba) instead to avoid that pain:

```
$ conda install -n base conda-forge::mamba
```

Create a new conda environment with most dependencies installed using mamba:

```
$ mamba create -n policy_analyzer -c anaconda -c conda-forge \
    'spacy==2.3.7' 'cudatoolkit=10.0' cupy cudnn \
    nccl 'requests==2.24.0' cython inflect boto3
```
Run the following instead if there is no GPU/CUDA:
```
$ mamba create -n policy_analyzer -c anaconda -c conda-forge \
    'spacy==2.3.7' 'requests==2.24.0' cython inflect boto3
```

Install [neuralcoref](https://github.com/huggingface/neuralcoref) library from source:

```
$ git clone git@github.com:huggingface/neuralcoref.git
```
If the SSH key on the local machine has not been set up yet, this command might throw the following error:
```
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.
```
We simply have to run the following `git clone` command instead:
```
git clone https://github.com/huggingface/neuralcoref.git
```
Next, run the following two commands:
```
$ cd neuralcoref
$ pip install -e .
```

Install spaCy's English NLP pipelines:

```
$ python -m spacy download en
$ python -m spacy download en_core_web_lg
```

## Usage

### Named-entity recognition (NER)

All code related to NER training are inside `ner` folder:

```
$ cd ner/
```

First step, use `gen_ner_data.py` to generate synthetic training data (10000 sentences for training and 2000 for validation):

```
$ python gen_ner_data.py train_dataset.json 10000
$ python gen_ner_data.py dev_dataset.json 2000
```

Train the NLP model:

```
$ python train.py
```

The model checkpoints are saved to `checkpoints/` folder. By default, the script stops after 30 epochs. Typically I use the last checkpoint (`checkpoints/29`).

### Analyzing Text

Scripts under `pattern` folder are for analyzing privacy policy text. Run `test_single.py` to analyze one text segment:

```
$ cd pattern/
$ python test_single.py ../ner/checkpoints/29
Text: If you browse our Websites, we may collect certain information about your use of our Websites. This information may include your IP address and geographical location.
'we' 'collect' 'certain information'
> 'certain information' INCLUDES ['your IP address', 'geographical location']
```

Here it extracts the tuple `('we', 'collect', 'certain information')` and resolves `certain information` to `your IP address` and  `geographical location`.

## Notes

The latest spaCy version is 3.x, which supports transformer-based models. Unfortunately, neuralcoref library [hasn't yet support spaCy 3](https://github.com/huggingface/neuralcoref/issues/295). So I had to use spaCy 2 and thus cannot use transformer models.

Here are some useful resources to understand the code:

- [spaCy v2 website](https://v2.spacy.io/)

- [How to Train spaCy to Autodetect New Entities (NER)](https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/)

- [neuralcoref demo](https://huggingface.co/coref/)
