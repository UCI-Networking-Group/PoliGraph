#!/usr/bin/env python3
import json
import random
import re
import sys

import inflect
import spacy
import tqdm


def load_list(fname):
    with open(fname) as fin:
        for line in fin:
            line = line.strip()
            if line and not line.startswith('#'):
                yield line


def main():
    outfile = sys.argv[1]
    n_samples = int(sys.argv[2])
    templates = list(load_list('template.txt'))

    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_trf')

    entities = []
    for ent in load_list('entities.txt'):
        ent = ent.replace('.', '')
        simp_ent = re.sub(r'\W*(Inc|Co|Corp|Corporation|Company|Ltd|Incorporated|LLC)$', '', ent, flags=re.I)
        entities.append(ent)

        if ent != simp_ent:
            entities.append(simp_ent)

    inflect_engine = inflect.engine()
    data_types = []

    for dtype in load_list('data_types.txt'):
        data_types.append(dtype)
        plural_form = inflect_engine.plural(dtype)

        if plural_form != dtype:
            data_types.append(plural_form)

        if ' number' in dtype:
            data_types.append(dtype.replace(' number', 'no.'))
        elif 'info' not in dtype:
            data_types.append(dtype + ' information')

    dataset = []

    for _ in tqdm.tqdm(range(n_samples)):
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

        doc = nlp(tpl)
        for ent in doc.ents:
            ent_label = (ent.start_char, ent.end_char, ent.label_)
            for lb in labels:
                if len(range(max(lb[0], ent_label[0]), min(lb[1], ent_label[1]) + 1)) > 0:
                    break
            else:
                if ent_label[-1] != 'ORDINAL':
                    labels.append(ent_label)

        dataset.append((tpl, {"entities": labels}))

    with open(outfile, "w") as fout:
        json.dump(dataset, fout)


if __name__ == "__main__":
    main()
