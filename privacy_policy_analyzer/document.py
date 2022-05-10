#!/usr/bin/env python3

import bisect
import enum
import json
import logging
import pickle
from functools import cached_property
from itertools import chain
from pathlib import Path

import networkx as nx
from regex import D
import spacy
from anytree import NodeMixin
from spacy import displacy
from spacy.tokens import Doc, Span
from unidecode import unidecode

from privacy_policy_analyzer.named_entity_recognition import (ACTOR_KEYWORDS, DATATYPE_KEYWORDS,
                                                              label_simple_noun_phrases)


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

        self.context = [s := self]
        while (s := s.parent) is not None:
            self.context.append(s)

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

        with open(self.workdir / "plaintext.txt", "w") as fout:
            fout.write(self.full_doc.text)

    def __init_doc(self, nlp):
        # first pass (NER -> noun chunks)
        self.noun_chunks = noun_chunks = []

        full_doc = self.get_full_doc(nlp)
        full_doc = nlp(full_doc)
        token_sources = full_doc.user_data["source"]

        label_simple_noun_phrases(full_doc)

        for ent in full_doc.ents:
            # exclude NER types that are not useful (e.g. PERCENT/DATE/LAW/LOC...)
            if ent.label_ not in {"NN", "DATA", "ACTOR", "EVENT", "FAC", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"}:
                continue

            # Rule-based NER completion
            if ent.root.lemma_.lower() in DATATYPE_KEYWORDS:
                label = "DATA"
            elif ent.root.lemma_.lower() in ACTOR_KEYWORDS:
                label = "ACTOR"
            elif ent.root.pos_ == "PRON" and ent.root.lemma_.lower() in {"i", "we", "you", "he", "she"}:
                label = "ACTOR"
            else:
                label = ent.label_

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

            noun_chunks.append((segment_id, left_token_index, right_token_index, label))

        # second pass (grouping chunks)
        full_doc = self.get_full_doc(nlp)
        token_sources = full_doc.user_data["source"]

        # apply tagger/parser but not NER
        for name, pipe in nlp.pipeline:
            if name in {"tok2vec", "transformer", "tagger", "parser", "attribute_ruler", "lemmatizer"}:
                full_doc = pipe(full_doc)

    @cached_property
    def default_nlp(self):
        return spacy.blank("en")

    def get_full_doc(self, nlp=None, apply_pipe=False):
        if nlp is None:
            nlp = self.default_nlp

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
                all_docs.append(Doc(nlp.vocab, words=prefix, spaces=[not c.isspace() for c in prefix]))
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
                all_docs.append(Doc(nlp.vocab, words=suffix, spaces=[not c.isspace() for c in suffix]))
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

    def build_doc(self, context, nlp, apply_pipe=False, load_ner=False):
        if any(s not in self.segments for s in context):
            raise ValueError("Unknown segment in the provided context")

        tokens = []
        spaces = []
        token_sources = []
        ent_positions = []
        previous_segment = None

        for s in reversed(context):
            # Propoerly concatenate segments
            if len(tokens) > 0:
                if previous_segment.segment_type is SegmentType.HEADING:
                    # Insert some linebreaks after a heading
                    tokens.extend(["\n", "\n"])
                    spaces.extend([False, False])
                    token_sources.extend([None, None])
                elif previous_segment.segment_type is SegmentType.LISTITEM:
                    # Add a colon before a LISTITEM
                    if tokens[-1].isalnum():
                        tokens.append(":")
                        spaces.append(True)
                        token_sources.append(None)
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
        src1 = token1._.src
        src2 = token2._.src

        if src1 is None or src2 is None:
            raise ValueError("Invalid token link")

        self.token_relationship.add_edge(src1, src2, relationship=relationship)

    def get_link(self, token1, token2):
        src1 = token1._.src
        src2 = token2._.src

        if src1 is None or src2 is None:
            raise ValueError("Invalid tokens")

        if edge_data := self.token_relationship.get_edge_data(src1, src2):
            return edge_data["relationship"]
        else:
            return None

    def get_all_links(self, token, direction=None):
        doc = token.doc
        source_rmap = doc.user_data["source_rmap"]

        match direction:
            case None | "out":
                edge_view = self.token_relationship.out_edges(token._.src, data=True)
            case "in":
                edge_view = self.token_relationship.in_edges(token._.src, data=True)
            case _:
                raise ValueError(f"Invalid direction: {direction}")

        for src, dst, data in edge_view:
            relationship = data["relationship"]
            src_dst_tokens = []

            for token_src in src, dst:
                try:
                    token = doc[source_rmap[token_src]]
                except KeyError:
                    full_doc = self.full_doc
                    rmap = full_doc.user_data["source_rmap"]
                    token = full_doc[rmap[token_src]]

                src_dst_tokens.append(token)

            yield src_dst_tokens[0], src_dst_tokens[1], relationship


def extract_segments_from_accessibility_tree(tree, tokenizer):
    """Process an accessibility tree into a list of DocumentSegment"""

    IGNORED_ELEMENTS = {"img", "image map", "button", "separator", "whitespace", "form",
                        "list item marker", "insertion", "diagram", "dialog", "tab",
                        "menu", "menubar", "internal frame", "listbox", "progressbar",
                        "alert", "button", "buttonmenu", "slider", "textbox"}
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
