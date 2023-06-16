#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict
import csv
import os

from poligrapher.graph_utils import KGraph, load_ontologies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    args = parser.parse_args()

    data_ontology, entity_ontology = load_ontologies()
    datatype_counters = [Counter(), Counter()]
    entity_counters = [Counter(), Counter()]
    possible_meanings_data = defaultdict(set)
    possible_meanings_entity = defaultdict(set)
    def_count = Counter()

    for d in args.workdirs:
        print(f"Processing {d} ...")

        kgraph_path = os.path.join(d, 'graph-original.yml')
        kgraph = KGraph(kgraph_path)

        common_datatypes = set(data_ontology.nodes) & set(kgraph.datatypes)
        common_entities = set(entity_ontology.nodes) & set(kgraph.entities)

        special_datatype_terms = set()
        special_entity_terms = set()
        defined_terms = set()

        for datatype in common_datatypes:
            for term in kgraph.ancestors(datatype):
                special_datatype_terms.add(term)
                possible_meanings_data[term].add(datatype)
                defined_terms.add(term)

        for entity in common_entities:
            for term in kgraph.ancestors(entity):
                special_entity_terms.update(term)
                possible_meanings_entity[term].add(entity)
                defined_terms.add(term)

        def_count.update(defined_terms)

        special_datatype_terms.difference_update(common_datatypes)
        special_entity_terms.difference_update(common_entities)

        datatype_counters[0].update(kgraph.datatypes)
        datatype_counters[1].update(special_datatype_terms)
        entity_counters[0].update(kgraph.entities)
        entity_counters[1].update(special_entity_terms)

    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["type", "term", "def_count", "use_count", "possible_meanings"])
        writer.writeheader()

        for term, counts in datatype_counters[0].most_common():
            if counts > 10 and term in datatype_counters[1]:
                writer.writerow({
                    "type": "DATA",
                    "term": term,
                    "def_count": def_count[term],
                    "use_count": counts,
                    "possible_meanings": "|".join(sorted(possible_meanings_data[term]))
                })

        for term, counts in entity_counters[0].most_common():
            if counts > 10 and term in datatype_counters[1]:
                writer.writerow({
                    "type": "ENTITY",
                    "term": term,
                    "def_count": def_count[term],
                    "use_count": counts,
                    "possible_meanings": "|".join(sorted(possible_meanings_entity[term]))
                })


if __name__ == "__main__":
    main()
