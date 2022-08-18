#!/usr/bin/env python3

from pathlib import Path
import json
import os
import sqlite3
from collections import deque
import networkx as nx
import argparse

DATATYPE_MAPPING = {
    "mac address": ["mac address"],
    "router ssid": ["router ssid"],
    "android id": ["android id"],
    "sim serial number": ["sim serial number"],
    "imei": ["imei"],
    "advertising identifier": ["advertising id"],
    "gsfid": ["gsf id"],
    "serial number": ["serial number"],
    "email address": ["email address"],
    "phone number": ["phone number"],
    "person name": ["person name"],
    "geographical location": ["geolocation", "coarse geolocation", "precise geolocation"],
}

DATATYPE_MAPPING_REVERSE = {}

def iter_hypernyms(graph, start_node):
    bfs_queue = deque()
    bfs_queue.append((start_node,))
    visited_nodes = {start_node}

    while len(bfs_queue) > 0:
        path = bfs_queue.popleft()
        yield path

        for parent, _, rel in graph.in_edges(path[-1], keys=True):
            if parent not in visited_nodes and rel == "SUBSUM":
                visited_nodes.add(parent)
                bfs_queue.append((parent,) + path)

for k, li in DATATYPE_MAPPING.items():
    for v in li:
        DATATYPE_MAPPING_REVERSE[v] = k

parser = argparse.ArgumentParser()
parser.add_argument("flow_json", help="PoliCheck flow JSON file")
parser.add_argument("privacy_policy_root", help="Input privacy policy directories")
parser.add_argument("policheck_ext", help="PoliCheck /ext folder")
args = parser.parse_args()

conn = sqlite3.connect(os.path.join(args.policheck_ext, "output", "db", "consistency_results_1.db"))
cursor = conn.cursor()

policy_root = Path(args.privacy_policy_root)
visited_privacy_policies = set()

count_common = 0
count_policheck_unique = 0
count_our_unique = 0

with open(args.flow_json) as fin:
    for app_id, info in json.load(fin).items():
        kgraph_path = policy_root / info["privacy_policy_id"] / 'graph.gml'

        if kgraph_path in visited_privacy_policies:
            continue
        else:
            visited_privacy_policies.add(kgraph_path)

        if not kgraph_path.exists():
            continue

        kgraph = nx.read_gml(kgraph_path)

        our_tuples = set()

        for data_type, policheck_data_type in DATATYPE_MAPPING_REVERSE.items():
            if data_type in kgraph:
                for parent, *_ in iter_hypernyms(kgraph, data_type):
                    for entity, _, rel in kgraph.in_edges(parent, keys=True):
                        if rel == "COLLECT":
                            negation = 0
                        elif rel == "NOT_COLLECT":
                            continue
                            negation = 1
                        else:
                            continue

                        if "we" in kgraph and nx.has_path(kgraph, "we", entity):
                            party = 1
                        else:
                            party = 3

                        our_tuples.add((DATATYPE_MAPPING_REVERSE[data_type], negation, party))

        policheck_tuples = set()
        policheck_sentence_list = {}

        cursor.execute("""SELECT appId,entity,collect,data,sentenceId FROM AppPolicySentences
            LEFT JOIN Policy ON AppPolicySentences.policyId == Policy.policyId
            WHERE appId = ?
        """, (app_id,))

        while db_row := cursor.fetchone():
            app_id, entity, action, data_type, sentence = db_row

            if data_type not in DATATYPE_MAPPING:
                continue

            if action == "collect":
                negation = 0
            else:
                continue
                negation = 1

            if entity == "we":
                party = 1
            else:
                party = 3

            t = (data_type, negation, party)
            policheck_tuples.add(t)

            if t not in policheck_sentence_list:
                policheck_sentence_list[t] = []

            policheck_sentence_list[t].append(sentence)

        if len(policheck_tuples) == 0 or len(our_tuples) == 0:
            continue

        count_common += len(our_tuples & policheck_tuples)
        count_our_unique += len(our_tuples - policheck_tuples)
        count_policheck_unique += len(policheck_tuples - our_tuples)

        #print(kgraph_path)
        #for t in policheck_tuples - our_tuples:
        #    print(t)
        #    for sent in policheck_sentence_list[t]:
        #        print(sent)
        #print(our_tuples - policheck_tuples)

    print("COMMON", count_common)
    print("OUR UNIQUE", count_our_unique)
    print("POLICHECK UNIQUE:", count_policheck_unique)

conn.close()
