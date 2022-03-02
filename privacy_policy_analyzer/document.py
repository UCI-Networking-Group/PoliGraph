#!/usr/bin/env python3
"""Process accessibility tree into a PolicyDocument"""

import enum
import json
import logging
import pickle
from itertools import chain
from pathlib import Path

import networkx as nx
import spacy
from anytree import NodeMixin
from spacy import displacy
from spacy.tokens import Doc, Span
from unidecode import unidecode


class SegmentType(enum.Enum):
    HEADING = enum.auto()
    LISTITEM = enum.auto()
    TEXT = enum.auto()


class DocumentSegment(NodeMixin):
    def __init__(self, segment_id, segment_type, doc, parent=None):
        super().__init__()

        self.segment_id = segment_id
        self.segment_type = segment_type
        self.tokens = [str(t) for t in doc]
        self.spaces = [t.whitespace_ != "" for t in doc]

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
    def __init__(self, workdir, nlp=None, use_cache=True):
        self.workdir = Path(workdir)
        self.token_relationship = nx.DiGraph()

        if use_cache and (self.workdir / "document.pickle").exists():
            with open(self.workdir / "document.pickle", "rb") as fin:
                self.segments, self.ner_labels = pickle.load(fin)
        else:
            with open(self.workdir / "accessibility_tree.json", encoding="utf-8") as fin:
                accessibility_tree = json.load(fin)

            self.segments = extract_segments_from_accessibility_tree(accessibility_tree, nlp.tokenizer)
            self.ner_labels = self.__init_ner_labels(nlp)

    def get_full_doc(self):
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

            spaces = [False] * len(prefix_tokens) + s.spaces + [False] * len(postfix_tokens)
            doc = Doc(nlp.vocab, words=prefix_tokens + s.tokens + postfix_tokens, spaces=spaces)

            for ent_start, ent_end, ent_label in ents:
                ent_start += len(prefix_tokens)
                ent_end += len(prefix_tokens)
                doc.set_ents([Span(doc, ent_start, ent_end, ent_label)], default="unmodified")

            all_docs.append(doc)

        return Doc.from_docs(all_docs)

    def render_ner(self):
        displacy.serve(self.get_full_doc(), style="ent")

    def save(self):
        with open(self.workdir / "document.pickle", "wb") as fout:
            pickle.dump((self.segments, self.ner_labels), fout, pickle.HIGHEST_PROTOCOL)

    def __init_ner_labels(self, nlp):
        ner_labels = []
        all_docs = []

        for s in self.segments:
            ner_labels.append([])
            all_docs.append(self.build_doc(s, nlp))

        for doc in nlp.pipe(all_docs):
            token_sources = doc.user_data["source"]

            for ent in doc.ents:
                ent_start = token_sources[ent.start]
                ent_end = token_sources[ent.end - 1]

                if ent_start[0] != ent_end[0]:
                    logging.warning("Entity %s crosses segment border. Skipped")
                    continue

                segment_id = ent_start[0]
                new_ent_spec = (ent_start[1], ent_end[1] + 1, ent.label_)
                for left, right, _ in ner_labels[segment_id]:
                    if len(range(max(left, new_ent_spec[0]), min(right, new_ent_spec[1]))) != 0:
                        break
                else:
                    ner_labels[segment_id].append(new_ent_spec)

        return ner_labels

    def build_doc(self, core_segment, nlp, with_context=True, apply_pipe=False, load_ner=False):
        if core_segment not in self.segments:
            raise ValueError("Unknown segment")

        if with_context:
            segments = core_segment.iter_context()
        else:
            segments = [core_segment]

        tokens = []
        spaces = []
        token_sources = []
        ent_positions = []
        previous_segment = None

        for s in segments:
            if previous_segment:
                if previous_segment.segment_type is SegmentType.HEADING:
                    tokens.extend(["\n", "\n"])
                    spaces.extend([False, False])
                    token_sources.extend([None, None])
                else:
                    spaces[-1] = True

            if load_ner:
                ent_offset = len(tokens)
                for ent_start, ent_end, ent_label in self.ner_labels[s.segment_id]:
                    ent_positions.append((ent_offset + ent_start, ent_offset + ent_end, ent_label))

            for idx, (tok, has_space) in enumerate(zip(s.tokens, s.spaces)):
                tokens.append(tok)
                spaces.append(has_space)
                token_sources.append((s.segment_id, idx))

            previous_segment = s

        doc = Doc(nlp.vocab, words=tokens, spaces=spaces)
        doc.user_data["document"] = self
        doc.user_data["source"] = token_sources

        if apply_pipe:
            old_tokenizer = nlp.tokenizer
            nlp.tokenizer = lambda x: x
            doc = nlp(doc)
            nlp.tokenizer = old_tokenizer

        if load_ner:
            doc.set_ents([Span(doc, s, e, l) for s, e, l in ent_positions], default="outside")

        return doc

    def link(self, token1, token2, relationship):
        doc1 = token1.doc
        doc2 = token2.doc

        token1_source = doc1.user_data["source"][token1.i]
        token2_source = doc2.user_data["source"][token2.i]

        self.token_relationship.add_edge(token1_source, token2_source, relationship=relationship)


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
