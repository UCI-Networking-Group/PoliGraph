#!/usr/bin/env python3

import enum
import itertools
import json
import logging
import pickle
from itertools import chain
from pathlib import Path
import re
import string

import networkx as nx
from anytree import NodeMixin
from spacy.tokens import Doc, DocBin
from spacy.language import Language
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


def assemble_raw_doc(context):
    tokens = []
    spaces = []
    token_sources = []
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

        for idx, (tok, has_space) in enumerate(zip(s.tokens, s.spaces)):
            tokens.append(tok)
            spaces.append(has_space)
            token_sources.append((s.segment_id, idx))

        previous_segment = s

    return {
        "words": tokens,
        "spaces": spaces,
        "user_data": {
            "source": token_sources,
            "source_rmap": {src: i for i, src in enumerate(token_sources) if src is not None}
        }
    }


class PolicyDocument:
    @classmethod
    def initialize(cls, workdir, nlp):
        obj = cls(flag=True)
        obj.workdir = Path(workdir)
        obj.token_relationship = nx.DiGraph()
        obj.nlp = nlp

        with open(obj.workdir / "accessibility_tree.json", encoding="utf-8") as fin:
            accessibility_tree = json.load(fin)

        obj.segments = extract_segments_from_accessibility_tree(accessibility_tree, nlp.tokenizer)

        docs = []

        for seg in obj.segments:
            if seg.segment_type == SegmentType.LISTITEM:
                continue

            for i in range(len(seg.context)):
                if seg.context[i].segment_type == SegmentType.LISTITEM:
                    continue

                raw_doc_params = assemble_raw_doc(seg.context[:i+1])
                raw_doc = Doc(nlp.vocab, **raw_doc_params)
                raw_doc.user_data["id"] = (seg.segment_id, i)

                docs.append(raw_doc)

        obj.all_docs = dict()

        for doc in nlp.pipe(docs):
            # See: https://github.com/explosion/spaCy/discussions/7486
            doc._.trf_data = None

            doc_id = doc.user_data["id"]
            obj.all_docs[doc_id] = doc

        return obj

    @classmethod
    def load(cls, workdir, nlp):
        obj = cls(flag=True)
        obj.workdir = Path(workdir)
        obj.nlp = nlp

        with open(obj.workdir / "document.pickle", "rb") as fin:
            (obj.token_relationship, obj.segments, docbin_bytes) = pickle.load(fin)

        serialized_docs = DocBin().from_bytes(docbin_bytes)
        obj.all_docs = dict()

        for doc in serialized_docs.get_docs(nlp.vocab):
            doc_id = doc.user_data["id"]
            obj.all_docs[doc_id] = doc

        return obj

    def __init__(self, **kwargs):
        if kwargs.get("flag") is not True:
            raise NotImplementedError("Don't call me directly")

        # Make linter happy
        self.workdir: Path
        self.all_docs: dict[tuple[int, int], Doc]
        self.token_relationship: nx.DiGraph
        self.segments: list[DocumentSegment]
        self.nlp: Language

    def save(self):
        serialized_docs = DocBin(store_user_data=True, docs=self.all_docs.values())

        with open(self.workdir / "document.pickle", "wb") as fout:
            pickle.dump((
                self.token_relationship,
                self.segments,
                serialized_docs.to_bytes(),
            ), fout, pickle.HIGHEST_PROTOCOL)

    def iter_docs(self):
        yield from self.all_docs.values()

    def get_doc_with_context(self, segment):
        for i in range(len(segment.context) - 1, 0, -1):
            if (segment.segment_id, i) in self.all_docs:
                return self.all_docs[(segment.segment_id, i)]

        return self.all_docs[(segment.segment_id, 0)]

    def get_doc_without_context(self, segment):
        return self.all_docs[(segment.segment_id, 0)]

    def get_token_with_src(self, src):
        segment = self.segments[src[0]]
        long_doc = self.get_doc_with_context(segment)
        rmap = long_doc.user_data["source_rmap"]
        return long_doc[rmap[src]]

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

        if token._.src is None:
            return

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
                    token = self.get_token_with_src(token_src)

                src_dst_tokens.append(token)

            yield src_dst_tokens[0], src_dst_tokens[1], relationship


def extract_segments_from_accessibility_tree(tree, tokenizer):
    """Process an accessibility tree into a list of DocumentSegment"""

    IGNORED_ELEMENTS = {"img", "image map", "button", "separator", "whitespace", "form",
                        "list item marker", "insertion", "diagram", "dialog", "tab",
                        "menu", "menubar", "internal frame", "listbox", "progressbar",
                        "alert", "button", "buttonmenu", "slider", "textbox",
                        "application", "details"}
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
        logging.info("New segment: %r, Parent: %r", seg, parent)
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

    def fix_non_html_lists():
        def list_bullet_generator(start):
            if start in "*->#":
                return itertools.cycle(start)

            if re.match(r"\b1\b", start):
                tpl = start.replace("1", "{0}")
                return map(tpl.format, itertools.count(1))

            if re.match(r"\ba\b", start):
                tpl = start.replace("a", "{0}")
                return map(tpl.format, itertools.cycle(string.ascii_lowercase))

            if re.match(r"\bA\b", start):
                tpl = start.replace("A", "{0}")
                return map(tpl.format, itertools.cycle(string.ascii_uppercase))

            return None

        need_renumber = False
        i = 1  # Intentionally skip the first one

        while i < len(segments):
            before_seg = segments[i - 1]

            if (segments[i].segment_type == SegmentType.TEXT and
                (bullet_iter := list_bullet_generator(segments[i].tokens[0])) is not None and
                before_seg.segment_type in [SegmentType.TEXT, SegmentType.HEADING] and
                len(before_seg.tokens) > 0 and before_seg.tokens[-1] == ":"):

                parent_seg = segments[i].parent
                j = i

                while (j < len(segments) and
                       segments[j].segment_type == SegmentType.TEXT and
                       segments[j].tokens[0] == next(bullet_iter) and
                       segments[j].parent == parent_seg):
                    j += 1

                num_segments = j - i

                if num_segments > 1:
                    logging.info("Convert text segments [%d, %d] to list items", i, j - 1)

                    for k in range(num_segments):
                        current_seg = segments[i + k * 2]

                        listitem_seg = DocumentSegment(0, SegmentType.LISTITEM, tokenizer(""), before_seg)
                        current_seg.parent = listitem_seg

                        if len(current_seg.tokens) > 1:
                            current_seg.tokens.pop(0)
                            current_seg.spaces.pop(0)

                        segments.insert(i + k * 2, listitem_seg)

                    need_renumber = True
                    i += k * 2 + 2
                else:
                    i += 1
            else:
                i += 1

        if need_renumber:
            logging.info("Re-number segment IDs")

            for idx, seg in enumerate(segments):
                seg.segment_id = idx

    iterate(tree)
    fix_non_html_lists()
    return segments
