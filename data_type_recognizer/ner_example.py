import spacy
from spacy.language import Language


@Language.component(
    "remove_unused_entities",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
)
def remove_unused_entities(doc):
    ents = []
    for e in doc.ents:
        if e.label_ not in ['ORDINAL']:
            ents.append(e)

    doc.set_ents(ents, default="outside")
    return doc


def main():
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    our_ner = spacy.load("output/model-best")

    our_ner.replace_listeners("transformer", "ner", ["model.tok2vec"])
    nlp.add_pipe(
        "remove_unused_entities",
        name="remove_unused_entities",
        after="ner",
    )
    nlp.add_pipe(
        "ner",
        name="ner_datatype",
        source=our_ner,
        after="remove_unused_entities",
    )

    line = input("Text: ")
    doc = nlp(line)

    for e in doc.ents:
        print(e.text, e.label_)


if __name__ == "__main__":
    main()
