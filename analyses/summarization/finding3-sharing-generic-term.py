import argparse
import os

import networkx as nx

from poligrapher.graph_utils import KGraph, load_ontologies

DATATYPE_CATEGORIES = [
    "software identifier",
    "hardware identifier",
    "geolocation",
    "government identifier",
    "contact information",
    "protected classification",
    "biometric information",
    "internet activity",
]

ENTITY_CATEGORIES = [
    "UNSPECIFIED_ACTOR",
    "advertiser",
    "analytic provider",
    "auth provider",
    "content provider",
    "email service provider",
    "social media",
]


def check_sharing_in_generic_terms(data_category_mapping, entity_ontology, kgraph):
    for data_type in kgraph.datatypes:
        covered_categories = set()

        for child in kgraph.descendants(data_type):
            if child in data_category_mapping:
                covered_categories.add(data_category_mapping[child])

        if len(covered_categories) <= 1:
            continue

        # Now we have a generic term that subsumes data types in more than 1 category
        collect_entities = set()

        for entity in kgraph.who_collect(data_type):
            # Skip first-party and unspecific actor
            if kgraph.subsum("we", entity):
                continue

            # Third-party category name
            if entity in ENTITY_CATEGORIES:
                collect_entities.add(entity)
                continue

            # Guess category based on the graph
            entity_cats = set()

            for anc in kgraph.ancestors(entity):
                if anc in ENTITY_CATEGORIES:
                    entity_cats.add(anc)

            # Guess category based on the ontology
            if len(entity_cats) == 0 and entity in entity_ontology.nodes():
                parents = {ent_cat for ent_cat, _ in entity_ontology.in_edges(entity)}
                parents.discard("UNKNOWN")

                if len(parents) == 1:
                    entity_cats.add(parents.pop())

            collect_entities.update(entity_cats)

        if collect_entities:
            return (collect_entities, data_type, covered_categories)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    args = parser.parse_args()

    data_ontology, entity_ontology = load_ontologies()
    data_category_mapping = {}

    for data_cat in DATATYPE_CATEGORIES:
        if data_cat != "UNSPECIFIED_DATA":
            data_category_mapping[data_cat] = data_cat

            for child in nx.descendants(data_ontology, data_cat):
                data_category_mapping[child] = data_cat

    result_count = 0

    for d in args.workdirs:
        kgraph_path = os.path.join(d, 'graph-original.yml')
        kgraph = KGraph(kgraph_path)

        if info := check_sharing_in_generic_terms(data_category_mapping, entity_ontology, kgraph):
            result_count += 1

            print(d)
            print(info)

    print(f"{result_count} policies declare third-party collection using generic terms")


if __name__ == "__main__":
    main()
