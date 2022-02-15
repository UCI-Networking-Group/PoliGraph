#!/usr/bin/env python3
"""Process accessibility tree into a PolicyDocument"""

import argparse
import enum
import json
import logging
from itertools import chain
from pathlib import Path

import spacy
from anytree import NodeMixin
from spacy import displacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from unidecode import unidecode


class SegmentType(enum.Enum):
    HEADING = enum.auto()
    LISTITEM = enum.auto()
    TEXT = enum.auto()


class DocumentSegment(NodeMixin):
    def __init__(self, segment_id, segment_type, tokens, parent=None):
        super().__init__()

        self.segment_id = segment_id
        self.segment_type = segment_type
        self.tokens = [str(t) for t in tokens]

        self.parent = parent

    def __repr__(self):
        return f"Segment #{self.segment_id}, type={self.segment_type.name}"

    def iter_context(self):
        s = self
        context = []

        while s is not None:
            context.append(s)
            s = s.parent

        yield from reversed(context)


class PolicyDocument:
    def __init__(self, workdir, nlp):
        self.workdir = Path(workdir)

        with open(self.workdir / "accessibility_tree.json", encoding="utf-8") as fin:
            accessibility_tree = json.load(fin)

        self.segments = extract_segments_from_accessibility_tree(accessibility_tree, nlp.tokenizer)
        self.ner_labels = PolicyDocument._init_ner_labels(self.segments, nlp)

    def render_ner(self):
        nlp = spacy.blank("en")
        all_docs = []

        for ents, s in zip(self.ner_labels, self.segments):
            prefix_tokens = []
            postfix_tokens = []

            if s.segment_type is SegmentType.HEADING:
                prefix_tokens.extend("\n\n#")
                postfix_tokens.extend("\n\n")
            elif s.segment_type is SegmentType.LISTITEM:
                postfix_tokens.append("*")
            else:
                postfix_tokens.append("\n")

            doc = Doc(nlp.vocab, words=prefix_tokens + s.tokens + postfix_tokens)

            for ent_start, ent_end, ent_label in ents:
                ent_start += len(prefix_tokens)
                ent_end += len(prefix_tokens)
                doc.set_ents([Span(doc, ent_start, ent_end, ent_label)], default="unmodified")

            all_docs.append(doc)

        combined_doc = Doc.from_docs(all_docs)
        displacy.serve(combined_doc, style="ent")

    @staticmethod
    def _init_ner_labels(segments, nlp):
        ner_labels = []
        all_docs = []

        for s in segments:
            ner_labels.append([])
            all_docs.append(PolicyDocument._build_doc_from_segments(s.iter_context(), nlp))

        for doc in nlp.pipe(all_docs):
            token_sources = doc.user_data["source"]

            for ent in doc.ents:
                ent_start = token_sources[ent.start]
                ent_end = token_sources[ent.end - 1]

                if ent_start[0] != ent_end[0]:
                    logging.warning("Entity %s crosses segment border. Skipped")
                    continue

                segment_id = ent_start[0]
                ner_labels[segment_id].append((ent_start[1], ent_end[1] + 1, ent.label_))

        return ner_labels

    @staticmethod
    def _build_doc_from_segments(segments, nlp):
        tokens = []
        token_ids = []
        previous_segment = None

        for s in segments:
            if previous_segment:
                if previous_segment.segment_type is SegmentType.HEADING:
                    tokens.extend(["\n", "\n"])
                    token_ids.extend([None, None])

            for idx, tok in enumerate(s.tokens):
                tokens.append(tok)
                token_ids.append((s.segment_id, idx))

            previous_segment = s

        doc = Doc(nlp.vocab, words=tokens)
        doc.user_data["source"] = token_ids

        return doc


def extract_segments_from_accessibility_tree(tree, tokenizer):
    IGNORED_ELEMENTS = {"img", "image map", "button", "separator", "whitespace", "list item marker", "insertion"}
    SECTION_ELEMENTS = {"document", "article", "landmark", "section", "blockquote"}
    TEXT_CONTAINER_ELEMENTS = {"paragraph", "text", "link", "statictext", "label", "text container", "text leaf"}
    TODO_ELEMENTS = {"table", "definitionlist"}  # TODO: parse tables and <dl>

    # NOTE:
    # heading: "heading"
    # list: "list", "listitem"
    # table: "table", "columnheader", "row", "cell"
    # definition list: "definitionlist", "term", "definition"

    heading_stack = [(-1, None)]
    segments = []

    def extract_text(node):
        if node["role"] in IGNORED_ELEMENTS:
            return

        if "children" in node:
            for child in node.get("children", []):
                yield from extract_text(child)
        else:
            yield unidecode(node["name"].strip())

    def new_segment(segment_type, text, parent):
        seg = DocumentSegment(len(segments), segment_type, tokenizer(text), parent)
        segments.append(seg)
        return seg

    def iterate(node):
        if node["role"] in TODO_ELEMENTS:
            return
        elif node["role"] in IGNORED_ELEMENTS:
            return
        elif node["role"] in SECTION_ELEMENTS:
            for child in node.get("children", []):
                iterate(child)
        elif node["role"] == "heading":
            level = node["level"]

            while heading_stack[-1][0] >= level:
                heading_stack.pop()

            parent = heading_stack[-1][1]
            heading = new_segment(SegmentType.HEADING, " ".join(extract_text(node)), parent)
            heading_stack.append((level, heading))
        elif node["role"] == "list":
            parent = segments[-1] if segments else None

            for child in node.get("children", []):
                # if child["role"] != "listitem":
                #     raise ValueError("Invalid child element of a list: " + child["role"])

                # Many HTML lists are illformed so this has to be tolerant
                if child["role"] in IGNORED_ELEMENTS:
                    continue

                listitem = new_segment(SegmentType.LISTITEM, "", parent)
                text_buffer = []

                for grandchild in chain(child.get("children", []), [None]):
                    if grandchild is None or grandchild["role"] == "list":
                        current_text = " ".join(text_buffer).strip()

                        if len(current_text) > 0:
                            new_segment(SegmentType.TEXT, current_text, listitem)
                            text_buffer.clear()

                        if grandchild:
                            iterate(grandchild)
                    else:
                        text_buffer.extend(extract_text(grandchild))
        elif node["role"] in TEXT_CONTAINER_ELEMENTS:
            parent = heading_stack[-1][1]
            text = " ".join(extract_text(node))
            if text:
                new_segment(SegmentType.TEXT, text, parent)
        else:
            raise ValueError(f"Invalid role: {node['role']}")

    iterate(tree)
    return segments


@Language.component(
    "remove_unused_entities",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
)
def remove_unused_entities(doc):
    ents = []
    for e in doc.ents:
        if e.label_ not in ["ORDINAL"]:
            ents.append(e)

    doc.set_ents(ents, default="outside")
    return doc


@Language.component(
    "remove_invalid_entities",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
)
def remove_invalid_entities(doc):
    ents = []
    for e in doc.ents:
        # discard invalid named entities
        if e.root.pos_ in ["NOUN", "PROPN"] and \
            e.root.dep_ in ["pobj", "dobj", "nsubj", "nsubjpass", "dative", "nmod", "poss", "conj", "appos"]:
            ents.append(e)

    doc.set_ents(ents, default="outside")
    return doc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", help="working directory")
    parser.add_argument("ner", help="NER model directory")
    args = parser.parse_args()

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    our_ner = spacy.load(args.ner)

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
    nlp.add_pipe(
        "remove_invalid_entities",
        name="remove_invalid_entities",
        after="ner_datatype",
    )

    document = PolicyDocument(args.workdir, nlp)
    document.render_ner()


if __name__ == "__main__":
    main()
