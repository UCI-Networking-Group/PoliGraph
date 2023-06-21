# USENIX Security 2023 Artifact Evaluation (Additional Experiments)

This document, as a supplement to the document [USENIX Security 2023 Artifact Evaluation](./usenix-artifact-evaluation.md), provides instructions on how to reproduce additional results in Sections 5.3 and 5.4 of our main paper.

## Additional Experiments

### E4: Contradiction Analysis

In this experiment, we use PoliGraph to identify "conflicting edges" as defined in Section 5.3 of the main paper.

Change to the `~/dataset` directory before proceeding:

```
$ cd ~/dataset
```

Please complete E0 before starting this experiment.

Step 1. As contradiction analysis is based on the extended version of PoliGraph, we need to re-run the `build_graph` script with the extensions:

```
$ python -m poligrapher.scripts.build_graph --variant extended dedup/*
2023-06-20 13:00:21,773 [INFO] Processing dedup/0003137f872072c4e797e5a6dd82864b20845fe559b37b756478c058e9c9c7ec ...
<more output omitted>
2023-06-20 13:16:05,170 [INFO] Nodes to remove: Google, VK, Yandex ...
```

This step takes approximately 15 minutes to complete. After this step, you will find a `graph-extended.yml`, containing the extended version of the PoliGraph, in each privacy policy's subdirectory.

Step 2. The analysis scripts needed for this experiment are in the `analyses/contradictions/` directory in the Git repo. Copy them to the current directory for convenience:

```
$ cp -Tr ~/poligraph/analyses/contradictions ./contradictions
```

Step 3. Run the script to reclassify PolicyLint contradictions:

```
$ python contradictions/reclassify-policylint-contradictions.py -r external/policylint-ext/ -o contradictions/policylint-reclassify.csv dedup/*
<more output omitted>
Number of unique PolicyLint tuples: 2555
```

#### Results

Step 3 outputs reclassification results in a CSV file `contradictions/policylint-reclassify.csv`. We may use `xsv` to count the number of each kind of result, which should match Table 7 of the main paper:

```
$ xsv search -s labels '\bINVALID\b' contradictions/policylint-reclassify.csv | xsv count
183
$ xsv search -s labels '\bDIFF_\w+\b' contradictions/policylint-reclassify.csv | xsv count
731
$ xsv search -s labels '\bDIFF_PURPOSES\b' contradictions/policylint-reclassify.csv | xsv count
114
$ xsv search -s labels '\bDIFF_SUBJECTS\b' contradictions/policylint-reclassify.csv | xsv count
121
$ xsv search -s labels '\bDIFF_ACTIONS\b' contradictions/policylint-reclassify.csv | xsv count
624
$ xsv search -s labels '\bONTOLOGY\b' contradictions/policylint-reclassify.csv | xsv count
441
$ xsv search -s labels '\bCONFLICT\b' contradictions/policylint-reclassify.csv | xsv count
211
```

### E5: Data Flow-to-Policy Consistency Analysis

In this experiment, we compare each app's data flows (i.e., the actual data collection practices observed in the network traffic) to the data collection statements inferred from its PoliGraph.

Change to the `~/dataset` directory before proceeding:

```
$ cd ~/dataset
```

Step 1. Run PoliGraph-er with the historical (2019) version of privacy policies from the PoliCheck dataset to create PoliGraphs for them:

```
$ python -m poligrapher.scripts.init_document wb2019/*
$ python -m poligrapher.scripts.run_annotators wb2019/*
$ python -m poligrapher.scripts.build_graph wb2019/*
```

Note that this step takes a similar amount of time to E0 (approximately 3 hours).

Step 2. The analysis scripts needed for this experiment are in the `analyses/flow-consistency/` directory in the Git repo. Copy them to the current directory for convenience:

```
$ cp -Tr ~/poligraph/analyses/flow-consistency ./flow-consistency
```

Step 3. Convert PoliCheck's dataset of data flows (`flow.json`) into a JSON file for easier processing:

```
$ python flow-consistency/convert_flow_csv.py external/policheck-flow-analysis-ext/data/flows.csv flow-consistency/dnscache.bin flow-consistency/policheck.json
```

Note that the script tries to reverse resolve IPs without domains in the dataset. It may take some time depending on the network condition, and the results are subject to the dynamics of DNS answers. For the ease of reproducibility, you may use our version of `policheck.json`:

```
$ cp external/policheck-flow-analysis.json flow-consistency/policheck.json
```

Step 4. Run the script to assess the consistency status of each data flow based on PoliGraphs, and output the results to a CSV file:

```
$ python flow-consistency/check-policheck-flow-consistency.py -p wb2019/ flow-consistency/policheck.json flow-consistency/result-poligraph.csv
```

Step 5. To reproduce Figure 9 in the main paper, we need to run [PoliCheck](https://github.com/benandow/PrivacyPolicyAnalysis) to obtain its flow-to-policy consistency results as well. For the ease of reproducibility, we provide PoliCheck's results directly in the directory `external/policheck-flow-analysis-ext/`.

Step 6. Generate figures of data flow-to-policy consistency results:

```
$ python flow-consistency/plot.py flow-consistency/result-poligraph.csv external/policheck-flow-analysis-ext/policheck_results.csv flow-consistency/figure.pdf
```

#### Results

Step 4 outputs the consistency status of each data flow based on PoliGraphs into `flow-consistency/result-poligraph.csv`. For example, the command below filters out data flows of the app `com.audionowdigital.player.frequencevie`:

```
$ xsv search -s app_id 'com.audionowdigital.player.frequencevie' flow-consistency/result-poligraph.csv
app_id,data_type,dest_entity,collection_consistency,policheck_consistency,purposes
com.audionowdigital.player.frequencevie,phone number,we,clear,clear,
com.audionowdigital.player.frequencevie,email address,we,clear,clear,services
```

The app had first-party data flows `(we, phone number)` and `(we, email address)`, both of which are clearly disclosed (`policheck_consistency` column). Additionally, the second flow has the purpose "services".

The PDF file `flow-consistency/figure.pdf` from Step 6 should reproduce Figure 9 in the main paper, and Figure 11 in [our extended paper](https://arxiv.org/abs/2210.06746).
