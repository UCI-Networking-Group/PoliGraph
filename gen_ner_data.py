#!/usr/bin/env python3

import json
import random
import re


def load_list(fname):
    with open(fname) as fin:
        for line in fin:
            line = line.strip()
            if line and not line.startswith('#'):
                yield line


def main():
    templates = list(load_list('template.txt'))

    entities = []
    for ent in load_list('entities.txt'):
        ent = ent.replace('.', '')
        simp_ent = re.sub(r'\W*(Inc|Co|Corp|Corporation|Company|Ltd|Incorporated)$', '', ent, flags=re.I)
        entities.append(ent)
        if ent != simp_ent:
            entities.append(simp_ent)

    data_types = []
    for dtype in load_list('data_types.txt'):
        data_types.append(dtype)

        if 'info' not in dtype:
            data_types.append(dtype + ' info')
            data_types.append(dtype + ' information')

    dataset = []

    for _ in range(10000):
        tpl = random.choice(templates)
        labels = []

        while True:
            m = re.search('{(?:DATA|ORG)}', tpl)
            if not m:
                break

            if m[0] == '{DATA}':
                rep = random.choice(data_types)
            elif m[0] == '{ORG}':
                rep = random.choice(entities)

            labels.append((m.span()[0], m.span()[0] + len(rep), m[0][1:-1]))
            tpl = tpl.replace(m[0], rep, 1)

        dataset.append((tpl, {"entities": labels}))

    with open("ner_dataset.json", "w") as fout:
        json.dump(dataset, fout)


if __name__ == "__main__":
    main()
