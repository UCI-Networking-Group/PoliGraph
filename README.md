# PoliGraph: Automated Privacy Policy Analysis using Knowledge Graphs

In USENIX 2023, we proposed PoliGraph, a framework to represent data collection statements in a privacy policy as a knowledge graph. We implemented an NLP-based tool, PoliGraph-er, to generate PoliGraphs and allow us to perform many analyses.

This repository is hosts the source code for PoliGraph, including:

- PoliGraph-er software - see instructions below.
- Evaluation scripts under `evals/`.
- PoliGraph analyses scripts under `analyses/`
- Dataset preparation scripts under `datasets/`.
- Model training scripts under `models/`.

## Citation

If you create a publication based on PoliGraph, please cite the paper as follows:

```bibtex
@inproceedings{cui2023poligraph,
  title     = {{PoliGraph: Automated Privacy Policy Analysis using Knowledge Graphs}},
  author    = {Cui, Hao and Trimananda, Rahmadi and Markopoulou, Athina and Jordan, Scott},
  booktitle = {Proceedings of the 32nd USENIX Security Symposium (USENIX Security 23)},
  year      = {2023}
}
```

## System Requirements

We test all the code in this repository on a server with the following configuration:
- CPU: Intel Xeon Silver 4316 (2 sockets x 20 cores x 2 threads)
- Memory: 512 GiB
- GPU: NVIDIA RTX A5000 (24 GiB of video memory)
- OS: Debian GNU/Linux 11 (bullseye)

A Linux server with 32 GiB of memory, 20 GiB of free disk space (after installing conda), and a similar NVIDIA GPU should suffice to run everything. Note that PoliGraph-er can run without a GPU, but the performance would be much worse.

## PoliGraph-er

PoliGraph-er is the software to generate PoliGraph from the text of a privacy policy.

### Installation

PoliGraph-er is written in Python. We use conda to manage the Python dependencies. Please follow [this instruction](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) to download and install conda first.

After cloning this repository, change the working directory to the cloned directory.

Create a new conda environment named `poligraph` with dependencies installed:

```sh
$ conda env create -n poligraph -f environment.yml
$ conda activate poligraph
```

Initialize Playwright library (used by the crawler script):

```sh
$ playwright install
```

Download `poligrapher-extra-data.tar.gz` from [here](https://drive.google.com/file/d/1qHifRx93EfTkg2x1e2W_lgQAgk7HcXhP/view?usp=sharing). Extract its content to `poligrapher/extra-data`:

```sh
$ tar xf /path/to/poligrapher-extra-data.tar.gz -C poligrapher/extra-data
```

Install the PoliGraph-er (`poligrapher`) library:

```sh
$ pip install --editable .
```

### Basic Usage

Here we illustrate how to generate a PoliGraph from a real privacy policy. We use the following privacy policy webpage as the example:

```sh
$ POLICY_URL="https://web.archive.org/web/20230330161225id_/https://proteygames.github.io/"
```

First, run the HTML crawler script to download the webpage:

```sh
$ python -m poligrapher.scripts.html_crawler ${POLICY_URL} example/
```

The directory `example/` will be used to store all the intermediate and final output associated with this privacy policy.

Second, run the `init_document` script to preprocess the webpage and run NLP pipeline on the privacy policy document:

```sh
$ python -m poligrapher.scripts.init_document example/
```

Third, execute the `run_annotators` script to run annotators on the privacy policy document.

```sh
$ python -m poligrapher.scripts.run_annotators example/
```

Last, the `build_graph` script to generate the PoliGraph:

```sh
$ python -m poligrapher.scripts.build_graph example/
```

The generated graph is stored at `example/graph-original.yml`.

### Batch Processing

If multiple directories are supplied to 

The `init_document`, `run_annotators` and `build_graph` scripts support batch processing. Simply supply multiple directories in the arguments:

```sh
$ python -m poligrapher.scripts.init_document dataset/policy1 dataset/policy2 dataset/policy3
$ python -m poligrapher.scripts.run_annotators dataset/policy1 dataset/policy2 dataset/policy3
$ python -m poligrapher.scripts.build_graph dataset/policy1 dataset/policy2 dataset/policy3
```

If all the subdirectories under `dataset` contain valid crawled webpages, you may simply supply `dataset/*` to let UNIX shell to expand the arguments.

### View PoliGraph

You may use a text editor to view `graph-original.yml`. The format is human-readable and fairly straightforward.

Alternatively, you can run `build_graph` with `--pretty` parameter:

```sh
$ python -m poligrapher.scripts.build_graph --pretty example/
```

This generates PoliGraph in the GraphML format (`example/graph-original.graphml`), which can be imported to [yEd](https://www.yworks.com/products/yed), a GUI graph editor.


## Other Scripts

We will update the documentation to explain the usage of other scripts.
