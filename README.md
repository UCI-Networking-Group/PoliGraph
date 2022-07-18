# Privacy Policy Analyzer

The NLP-based privacy policy analyzer.

## Dependencies

TODO: There can be some residues of my development environment in this long dependency list. Clean it up before we publish.

Create a new conda environment with most dependencies installed:

```
$ conda create --experimental-solver=libmamba -n nlp20220718 \
    'anaconda::python>=3.10' anaconda::numpy anaconda::pandas pytorch::pytorch \
    anaconda::pyyaml anaconda::lxml anaconda::bs4 anaconda::networkx \
    huggingface::transformers 'conda-forge::spacy>=3.4.0' \
    conda-forge::spacy-transformers  conda-forge::inflect conda-forge::anytree \
    conda-forge::tldextract conda-forge::requests-cache \
    conda-forge::unidecode microsoft::playwright conda-forge::langdetect \
    anaconda::werkzeug anaconda::ipykernel anaconda::pylint \
    anaconda::cudatoolkit=11.3 conda-forge::cudnn=8.2.1.32 conda-forge::cupy
```

If there is no GPU/CUDA, remove the last line from above command.

Install spaCy's English NLP pipelines:

```
$ python -m spacy download en_core_web_trf
$ python -m spacy download en_core_web_lg
$ python -m spacy download en_core_web_md
$ python -m spacy download en_core_web_sm
```

## Usage

Before performing the following commands, download this repository first to local machine:

```
$ git clone https://github.uci.edu/NetworkingGroup/privacy_policy_analyzer.git
```

### Named-entity recognition (NER)

All code related to NER training are inside `ner` folder:

```
$ cd data_type_recognizer/
```

First step, use `get_actor_entity_list.py` to generate `actor_entities.list`, a list of entity names from Wikidata:

```
$ python get_actor_entity_list.py
```

Run `gen_ner_data.py` generate synthetic training data (`train_dataset.spacy` and `dev_dataset.spacy`):

```
$ python gen_ner_data.py
```

Train the NLP model:

```
$ python -m spacy init fill-config ./base_config.cfg ./config.cfg
$ python -m spacy train config.cfg --output ./checkpoints --paths.train ./train_dataset.spacy --paths.dev ./dev_dataset.spacy --gpu-id 0
```

The model checkpoints are saved to `checkpoints` folder.

Now we go back to the root folder and run `privacy_policy_analyzer.scripts.repack_model` to embed our NER model into spaCy's transformer pipeline:

```
$ cd ..
$ python -m privacy_policy_analyzer.scripts.repack_model ner/checkpoints/model-best nlp_model
```

The analyzer will use the packed NLP pipeline stored in the `nlp_model` folder.


### Analyzing Text

TODO

## Notes

The latest spaCy version is 3.x, which supports transformer-based models. Unfortunately, neuralcoref library [hasn't yet support spaCy 3](https://github.com/huggingface/neuralcoref/issues/295). So I had to use spaCy 2 and thus cannot use transformer models.

Here are some useful resources to understand the code:

- [spaCy v2 website](https://v2.spacy.io/)

- [How to Train spaCy to Autodetect New Entities (NER)](https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/)

- [neuralcoref demo](https://huggingface.co/coref/)
