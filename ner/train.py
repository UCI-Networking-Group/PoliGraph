import json
import os
import random

import spacy
from spacy.training.example import Example
from spacy.util import compounding, minibatch


def main():
    spacy.prefer_gpu()

    nlp = spacy.load('en_core_web_sm')
    ner = nlp.get_pipe("ner")
    # Or from empty model:
    # nlp = spacy.blank("en")
    # nlp.add_pipe('ner')

    ner = nlp.get_pipe('ner')
    ner.add_label('ORG')
    ner.add_label('DATA')

    with open('train_dataset.json') as f:
        data = json.load(f)
        TRAIN_DATA = []
        for text, annotations in data:
            doc = nlp.make_doc(text)
            TRAIN_DATA.append(Example.from_dict(doc, annotations))

    with open('dev_dataset.json') as f:
        data = json.load(f)
        DEV_DATA = []
        for text, annotations in data:
            doc = nlp.make_doc(text)
            DEV_DATA.append(Example.from_dict(doc, annotations))

    with open("noise.txt") as f:
        for line in f:
            doc = nlp.make_doc(line.strip())
            labels = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
            TRAIN_DATA.append(Example.from_dict(doc, {"entities": labels}))

    os.makedirs("checkpoints", exist_ok=True)

    for iteration in range(30):
        # TRAINING
        losses = {}
        random.shuffle(TRAIN_DATA)

        batches = spacy.util.minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))

        with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'ner']):
            for batch in batches:
                nlp.update(batch, losses=losses, drop=0.3)

            print("Iteration", iteration)
            print("Losses", losses)

            # VALIDATION
            scores = nlp.evaluate(DEV_DATA)
            print(scores)

        # CHECKPOINT
        nlp.to_disk(os.path.join("checkpoints", str(iteration)))


if __name__ == "__main__":
    main()
