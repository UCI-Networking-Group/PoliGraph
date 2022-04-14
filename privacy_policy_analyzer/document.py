#!/usr/bin/env python3

import bisect
import enum
import json
import logging
import pickle
import re
from itertools import chain
from pathlib import Path

import networkx as nx
import spacy
from anytree import NodeMixin
from spacy import displacy
from spacy.tokens import Doc, Span
from unidecode import unidecode

from privacy_policy_analyzer.utils import expand_token


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

    @property
    def heading_level(self):
        if self.segment_type != SegmentType.HEADING:
            return None

        count = 0
        s = self
        while s is not None:
            if s.segment_type == SegmentType.HEADING:
                count += 1
            s = s.parent

        return count

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
                (
                    self.segments,
                    self.noun_chunks,
                    self.token_relationship,
                ) = pickle.load(fin)
        else:
            with open(self.workdir / "accessibility_tree.json", encoding="utf-8") as fin:
                accessibility_tree = json.load(fin)

            self.segments = extract_segments_from_accessibility_tree(accessibility_tree, nlp.tokenizer)
            self.__init_doc(nlp)

        self.full_doc = self.get_full_doc()

    def render_ner(self):
        displacy.serve(self.get_full_doc(), style="ent")

    def save(self):
        with open(self.workdir / "document.pickle", "wb") as fout:
            pickle.dump((
                self.segments,
                self.noun_chunks,
                self.token_relationship,
            ), fout, pickle.HIGHEST_PROTOCOL)

    def __init_doc(self, nlp):
        def label_unknown_noun_chunks(token):
            if (token.ent_iob_ == 'O'  # not in a named entity
                and token.pos_ in ["NOUN", "PRON", "PROPN"]  # noun or pronoun
                and re.search(r"[a-zA-Z]", token.text) is not None):  # ignore puncts due to bad tagging

                chunk = expand_token(token)

                if chunk.root.lemma_ in ["information", "datum"]:
                    chunk_type = "DATA"
                else:
                    chunk_type = "NN"

                ent = Span(token.doc, chunk.start, chunk.end, chunk_type)
                token.doc.set_ents([ent], default="unmodified")

            for child in token.children:
                label_unknown_noun_chunks(child)

        # first pass (NER -> noun chunks)
        self.noun_chunks = noun_chunks = []

        full_doc = self.get_full_doc(nlp)
        full_doc = nlp(full_doc)
        token_sources = full_doc.user_data["source"]

        for sent in full_doc.sents:
            label_unknown_noun_chunks(sent.root)

        for ent in full_doc.ents:
            # exclude NER types that are not useful (e.g. CARDINAL/PERCENT/DATE...)
            if ent.label_ not in {"NN", "DATA", "LAW", "EVENT", "FAC", "LOC",
                                  "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"}:
                continue

            # find the beginning and end of a chunk within a segment
            first_source = last_source = None

            i = ent.start
            while first_source is None:
                first_source = token_sources[i]
                i += 1

            i = ent.end
            while last_source is None:
                last_source = token_sources[i - 1]
                i -= 1

            if first_source[0] != last_source[0]:
                logging.warning("Chunk %r spans multiple segments. Skipped.", ent)
                continue

            segment_id = first_source[0]
            left_token_index = first_source[1]
            right_token_index = last_source[1] + 1

            noun_chunks.append((segment_id, left_token_index, right_token_index, ent.label_))

        # second pass (grouping chunks)
        full_doc = self.get_full_doc(nlp)
        token_sources = full_doc.user_data["source"]

        # apply tagger/parser but not NER
        for name, pipe in nlp.pipeline:
            if name in {"tok2vec", "transformer", "tagger", "parser", "attribute_ruler", "lemmatizer"}:
                full_doc = pipe(full_doc)

    def get_full_doc(self, nlp=None, apply_pipe=False):
        if nlp is None:
            nlp = spacy.blank("en")

        all_docs = []
        token_sources = []
        chunk_id = 0

        for s in self.segments:
            # TODO: This should be consistent with build_doc
            if s.segment_type is SegmentType.HEADING:
                prefix = ["\n\n", "#" * s.heading_level]
                suffix = ["\n\n"]
            elif s.segment_type is SegmentType.LISTITEM:
                prefix = ["*"]
                suffix = []
            else:
                prefix = []
                suffix = ["\n"]

            if len(prefix) > 0:
                all_docs.append(Doc(nlp.vocab, words=prefix, spaces=[False] * len(prefix)))
                token_sources.extend([None] * len(all_docs[-1]))

            doc = Doc(nlp.vocab, words=s.tokens, spaces=s.spaces)

            ents = []
            while chunk_id < len(self.noun_chunks):
                segment_id, left, right, label = self.noun_chunks[chunk_id]
                if segment_id != s.segment_id:
                    break

                span = Span(doc, left, right, label=label)
                ents.append(span)

                chunk_id += 1

            doc.set_ents(ents, default="outside")
            all_docs.append(doc)
            token_sources.extend((s.segment_id, i) for i in range(len(doc)))

            if len(suffix) > 0:
                all_docs.append(Doc(nlp.vocab, words=suffix, spaces=[False] * len(suffix)))
                token_sources.extend([None] * len(all_docs[-1]))

        if len(all_docs) > 0:
            full_doc = Doc.from_docs(all_docs, ensure_whitespace=False)
        else:
            # TODO: handle empty document
            full_doc = Doc(nlp.vocab, words=[" "])
            token_sources.append(None)

        full_doc.user_data["document"] = self
        full_doc.user_data["source"] = token_sources
        full_doc.user_data["source_rmap"] = {src: i for i, src in enumerate(token_sources) if src is not None}

        if apply_pipe:
            for name, pipe in nlp.pipeline:
                if name in {"tok2vec", "transformer", "tagger", "parser", "attribute_ruler", "lemmatizer"}:
                    full_doc = pipe(full_doc)

        return full_doc

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
        list_count = 0

        for s in segments:
            # Propoerly concatenate segments
            if len(tokens) > 0:
                if s.segment_type == SegmentType.LISTITEM:
                    # Add a colon or space before a LISTITEM
                    if tokens[-1].isalnum():
                        tokens.append(":")
                        spaces.append(True)
                        token_sources.append(None)
                    else:
                        spaces[-1] = True

                    list_count += 1
                elif previous_segment.segment_type is SegmentType.HEADING:
                    # Insert some linebreaks after a heading
                    tokens.extend(["\n", "\n"])
                    spaces.extend([False, False])
                    token_sources.extend([None, None])
                elif previous_segment.segment_type is SegmentType.LISTITEM:
                    tokens.extend(["\n", "*" * list_count])
                    spaces.extend([False, True])
                    token_sources.extend([None, None])
                else:
                    # Otherwise, just insert a space
                    spaces[-1] = True

            if load_ner:
                ent_offset = len(tokens)
                chunk_id = bisect.bisect_left(self.noun_chunks, (s.segment_id, 0, 0, ""))

                while chunk_id < len(self.noun_chunks):
                    segment_id, left, right, label = self.noun_chunks[chunk_id]
                    if segment_id == s.segment_id:
                        ent_positions.append((chunk_id, ent_offset + left, ent_offset + right, label))
                        chunk_id += 1
                    else:
                        break

            for idx, (tok, has_space) in enumerate(zip(s.tokens, s.spaces)):
                tokens.append(tok)
                spaces.append(has_space)
                token_sources.append((s.segment_id, idx))

            previous_segment = s

        doc = Doc(nlp.vocab, words=tokens, spaces=spaces)
        doc.user_data["document"] = self
        doc.user_data["source"] = token_sources
        doc.user_data["source_rmap"] = {src: i for i, src in enumerate(token_sources) if src is not None}

        if apply_pipe:
            old_tokenizer = nlp.tokenizer
            nlp.tokenizer = lambda x: x
            doc = nlp(doc)
            nlp.tokenizer = old_tokenizer

        if load_ner:
            ents = []
            for idx, left, right, label in ent_positions:
                span = Span(doc, left, right, label=label)
                ents.append(span)

            doc.set_ents(ents, default="outside")

        return doc

    def link(self, token1, token2, relationship):
        self.token_relationship.add_edge(token1._.src, token2._.src, relationship=relationship)

    def get_links(self, token):
        doc = token.doc
        source_rmap = doc.user_data["source_rmap"]

        for _, dest_source, data in self.token_relationship.out_edges(token._.src, data=True):
            relationship = data["relationship"]

            try:
                dest_token = doc[source_rmap[dest_source]]
            except KeyError:
                full_doc = self.full_doc
                rmap = full_doc.user_data["source_rmap"]
                dest_token = full_doc[rmap[dest_source]]

            yield dest_token, relationship


def extract_segments_from_accessibility_tree(tree, tokenizer):
    """Process an accessibility tree into a list of DocumentSegment"""

    IGNORED_ELEMENTS = {"img", "image map", "button", "separator", "whitespace",
                        "list item marker", "insertion", "diagram", "dialog", "tab"}
    SECTION_ELEMENTS = {"document", "article", "landmark", "section", "blockquote", "group",
                        "tablist", "tabpanel", "region"}
    TEXT_CONTAINER_ELEMENTS = {"paragraph", "text", "link", "statictext", "label", "text container", "text leaf"}
    TODO_ELEMENTS = {"table", "definitionlist", "figure"}  # TODO: parse tables and <dl>

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
                # Ideally a listitem child should be here
                # but many HTML lists are illformed so this has to be tolerant
                if child["role"] in IGNORED_ELEMENTS:
                    continue

                listitem = None
                text_buffer = []

                for grandchild in chain(child.get("children", []), [None]):
                    if grandchild is None or grandchild["role"] == "list":
                        current_text = " ".join(text_buffer).strip()

                        if len(current_text) > 0:
                            if listitem is None:
                                # Lazy creation to avoid empty LISTITEM
                                listitem = new_segment(SegmentType.LISTITEM, "", parent)

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
