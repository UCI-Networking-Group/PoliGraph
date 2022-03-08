#!/usr/bin/env python3
"""Process accessibility tree into a PolicyDocument"""

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

from privacy_policy_analyzer.utils import expand_token, get_conjuncts


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


class TokenGroupStorage:
    def __init__(self):
        self.ent_id_to_group = dict()
        self.group_to_ent_ids = dict()
        self.group_count = 0

    def __getitem__(self, group_id):
        for ent_id in self.group_to_ent_ids[group_id]:
            yield ent_id

    def get_group(self, token):
        ent_id = token._.ent_id
        return self.ent_id_to_group.get(ent_id)

    def create_group(self, token_list):
        new_group_id = self.group_count
        self.group_count += 1
        self.group_to_ent_ids[new_group_id] = members = set()

        for t in token_list:
            ent_id = t._.ent_id
            members.add(ent_id)
            self.ent_id_to_group[ent_id] = new_group_id

    def merge_groups(self, group1, group2):
        if group1 == group2:
            return

        group1_members = self.group_to_ent_ids[group1]

        for ent_id in self.group_to_ent_ids.pop(group2):
            self.ent_id_to_group[ent_id] = group1
            group1_members.add(ent_id)


class PolicyDocument:
    def __init__(self, workdir, nlp=None, use_cache=True):
        self.workdir = Path(workdir)

        self.token_groups = TokenGroupStorage()
        self.token_relationship = nx.DiGraph()

        if use_cache and (self.workdir / "document.pickle").exists():
            with open(self.workdir / "document.pickle", "rb") as fin:
                (
                    self.segments,
                    self.noun_chunks,
                    self.token_groups,
                    self.token_relationship,
                ) = pickle.load(fin)
        else:
            with open(self.workdir / "accessibility_tree.json", encoding="utf-8") as fin:
                accessibility_tree = json.load(fin)

            self.segments = extract_segments_from_accessibility_tree(accessibility_tree, nlp.tokenizer)
            self.__init_doc(nlp)

    def render_ner(self):
        displacy.serve(self.get_full_doc(), style="ent")

    def save(self):
        with open(self.workdir / "document.pickle", "wb") as fout:
            pickle.dump((
                self.segments,
                self.noun_chunks,
                self.token_groups,
                self.token_relationship,
            ), fout, pickle.HIGHEST_PROTOCOL)

    def __init_doc(self, nlp):
        def label_unknown_noun_chunks(token):
            if (token.ent_iob_ not in 'BI'  # not in a named entity
                and (token.pos_ in ["NOUN", "PROPN"] or token.tag_ == 'PRP')  # noun or pronoun
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
            if ent.label_ not in {"NN", "DATA", "LAW", "EVENT", "FAC", "LOC", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"}:
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

        for ent in full_doc.ents:
            if self.token_groups.get_group(ent.root) is not None:
                continue

            new_group = [ent.root]
            #print(group_id, end=": ")
            #print(ent, end=" | ")

            for conjunct in get_conjuncts(ent.root):
                if conjunct._.ent is None:
                    continue

                #print(conjunct, end=" | ")

                new_group.append(conjunct)

            #print()
            self.token_groups.create_group(new_group)

    def get_full_doc(self, nlp=None):
        if nlp is None:
            nlp = spacy.blank("en")

        all_docs = []
        token_sources = []
        noun_chunk_mapping = dict()
        noun_chunk_from_id = dict()
        chunk_id = 0

        for s in self.segments:
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
                noun_chunk_mapping[left + len(token_sources)] = chunk_id
                noun_chunk_from_id[chunk_id] = left + len(token_sources)

                chunk_id += 1

            try:
                doc.set_ents(ents, default="outside")
            except AttributeError:
                doc.ents = ents

            all_docs.append(doc)
            token_sources.extend((s.segment_id, i) for i in range(len(doc)))

            if len(suffix) > 0:
                all_docs.append(Doc(nlp.vocab, words=suffix, spaces=[False] * len(suffix)))
                token_sources.extend([None] * len(all_docs[-1]))

        full_doc = Doc.from_docs(all_docs, ensure_whitespace=False)
        full_doc.user_data["document"] = self
        full_doc.user_data["source"] = token_sources
        full_doc.user_data["noun_chunk"] = noun_chunk_mapping
        full_doc.user_data["noun_chunk_from_id"] = noun_chunk_from_id
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

        for s in segments:
            if previous_segment:
                if previous_segment.segment_type is SegmentType.HEADING and s.segment_type != SegmentType.LISTITEM:
                    tokens.extend(["\n", "\n"])
                    spaces.extend([False, False])
                    token_sources.extend([None, None])
                else:
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
        doc.user_data["noun_chunk"] = dict()
        doc.user_data["noun_chunk_from_id"] = dict()

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
                doc.user_data["noun_chunk"][left] = idx
                doc.user_data["noun_chunk_from_id"][idx] = left

            try:
                doc.set_ents(ents, default="outside")
            except AttributeError:
                doc.ents = ents

        return doc

    def group(self, token1, token2):
        group1 = self.token_groups.get_group(token1)
        group2 = self.token_groups.get_group(token2)

        if group1 is None or group2 is None:
            raise RuntimeError("invalid token")

        self.token_groups.merge_groups(group1, group2)

    def get_groupped_chunks(self, token):
        doc = token.doc
        noun_chunk_from_id = doc.user_data["noun_chunk_from_id"]
        group = self.token_groups.get_group(token)

        for linked_chunk_id in self.token_groups[group]:
            left = noun_chunk_from_id[linked_chunk_id]
            yield doc[left]._.ent

    def link(self, token1, token2, relationship):
        e1 = token1._.ent_id
        e2 = token2._.ent_id

        if e1 is None or e2 is None:
            raise RuntimeError("invalid token")

        self.token_relationship.add_edge(e1, e2, relationship=relationship)

    def get_links(self, token):
        doc = token.doc
        noun_chunk_from_id = doc.user_data["noun_chunk_from_id"]
        chunk_id = token._.ent_id

        if chunk_id not in self.token_relationship:
            return

        for _, dest_chunk_id, data in self.token_relationship.out_edges(chunk_id, data=True):
            # FIXME: Links should be made between tokens instead of noun chunks in the future
            relationship = data["relationship"]
            dest_token = doc[noun_chunk_from_id[dest_chunk_id]]
            yield dest_token, relationship


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
