# Privacy Policy Analyzer

The NLP-based privacy policy analyzer.


## Setup

### Dependencies

**Hardware**: An NVIDIA GPU with CUDA support and at least 8GB memory is recommended. All the instructions are tested with RTX A5000 GPU.

**System**: All the instructions are tested on Debian 11. Any modern GNU/Linux distribution should be sufficient.

We use conda to manage the Python environment. Please follow [this instruction](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) to download and install conda.

Create a new conda environment with dependencies installed:

```sh
$ conda env create -n nlp -f environment.yml
$ conda activate nlp
```

Playwright library (used by the crawler script) needs to be initialized:

```sh
$ playwright install
```

### Cloning This Repository

Use `git` to download this repository to local machine:

```sh
$ git clone https://github.uci.edu/NetworkingGroup/privacy_policy_analyzer.git
```

Clone external datasets as git submodules:

```sh
$ git submodule init
$ git submodule update
```


## NLP Pipeline Setup

### Pretrained NLP Pipelines

Install spaCy's English NLP pipelines:

```sh
$ python -m spacy download en_core_web_trf
$ python -m spacy download en_core_web_md
$ python -m spacy download en_core_web_sm
```

### Custom NER Training

Note that you may skip this step and use our released model checkpoint instead (see the next step).

All code related to NER training are inside `custom_ner_training` folder:

```sh
$ cd custom_ner_training/
```

Run `get_actor_entity_list.py` to fetch a list of entity names from Wikidata:

```sh
$ python get_actor_entity_list.py
# Output: actor_entities.list
```

Run `gen_ner_data.py` generate synthetic training data:

```sh
$ python gen_ner_data.py
# Output: train_dataset.spacy, dev_dataset.spacy
```

Train the custom NER model:

```sh
$ python -m spacy init fill-config base_config.cfg config.cfg
# Output: config.cfg
$ python -m spacy train config.cfg --output checkpoints/ --paths.train train_dataset.spacy --paths.dev dev_dataset.spacy --gpu-id 0
# Output: checkpoints/ folder
```

The model checkpoint with best performance is saved to `checkpoints/model-best`.

### Packing the NLP Pipeline

Run the `repack_model` script to embed the custom NER model into spaCy's NLP pipeline.

If you trained the NER model following the [Custom NER Training](#custom-ner-training) section, run the following commands:

```sh
$ cd ..
$ python -m privacy_policy_analyzer.scripts.repack_model custom_ner_training/checkpoints/model-best/ nlp_model/
# Output: nlp_model/ folder (about 1GB size)
```

Otherwise, please download our NER model checkpoint [here](#TODO) to `custom-ner-model.tar.gz` and run:

```sh
$ tar xf custom-ner-model.tar.gz
$ python -m privacy_policy_analyzer.scripts.repack_model model-best/ nlp_model/
# Output: nlp_model/ folder (about 1GB size)
$ rm -rf model-best/
```

The analyzer will use the packed NLP pipeline stored in the `nlp_model` folder.


## Privacy Policy Analyzer

We will use [KAYAK privacy policy](https://www.kayak.com/privacy) as an example.

### Download the Privacy Policy

Run the HTML crawler script to download the privacy policy:

```sh
$ mkdir examples/
$ python -m privacy_policy_analyzer.scripts.html_crawler https://www.kayak.com/privacy examples/kayak/
```

A successful run outputs something like:

```
2022-10-18 13:25:06,858 [INFO] Testing URL 'https://www.kayak.com/privacy' with HEAD request
2022-10-18 13:25:08,693 [INFO] Navigating to 'https://www.kayak.com/privacy'
2022-10-18 13:25:13,408 [INFO] Saved to examples/kayak
```

The `examples/kayak/` folder will be used by all the subsequent scripts to store intermediate data and results.

### HTML Preprocessing and NLP

Execute the `init_document` script to preprocess HTML and run NLP pipeline on the privacy policy document:

```sh
$ python -m privacy_policy_analyzer.scripts.init_document --nlp nlp_model examples/kayak
```

The script and subsequent scripts generally take the following arguments:
- The `--nlp` argument specifies the path to the packed NLP pipeline.
- Positional arguments (`examples/kayak`) specify one or more paths to privacy policies downloaded by the crawler script.

If you want to run the analyzer on a batch of privacy policies, it is recommended to run each script step once with all the paths appended in the command instead of multiple runs, because loading the NLP pipeline takes time.

### Annotators

Execute the `run_annotators` script to run annotators on the privacy policy document.

```sh
$ python -m privacy_policy_analyzer.scripts.run_annotators --nlp nlp_model
```

### Graph Building

The first time you run this step, please generate `entity_info.json` from the Tracker Radar and Crunchbase datasets:

```sh
$ python dataset_preprocessing/merge_tracker_radar_and_crunchbase.py external_datasets/tracker-radar external_datasets/crunchbase-data entity_info.json
```

The output file `entity_info.json` contains information that help to normalize company names and build the global entity ontology.

Execute the `build_graph` script to generate the PoliGraph:

```sh
$ python3 -m privacy_policy_analyzer.scripts.build_graph --nlp nlp_model -p privacy_policy_analyzer/extra-data/phrase_map.yml -e entity_info.json examples/kayak
```

The output is `examples/kayak/graph_trimmed.gml`. You may use a graph viewer like [yEd](https://www.yworks.com/products/yed) to open the file.
