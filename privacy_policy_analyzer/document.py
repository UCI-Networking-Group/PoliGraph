"""Module of privacy policy document representation"""

import enum
import io
import itertools
import json
import logging
import pickle
import re
import string
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import networkx as nx
import regex
from anytree import NodeMixin, RenderTree
from langdetect import DetectorFactory, detect_langs, lang_detect_exception
from spacy.language import Language
from spacy.tokens import Doc, DocBin, Token
from unidecode import unidecode

# Latin-1 Supplement / Latin Extended-A / Latin Extended-B + few exceptions
NON_ENGLISH_ALPHA = {chr(i) for i in range(0x00c0, 0x0250)} - {"é", "×", "÷"}
NON_ENGLISH_RE = regex.compile(
    r'[\p{Han}\p{Hangul}\p{Hiragana}\p{Katakana}\p{Armenian}\p{Arabic}\p{Cyrillic}\p{Greek}' +
    "".join(NON_ENGLISH_ALPHA) + "]")
# Stablize langdetect results: https://github.com/Mimino666/langdetect/issues/3
DetectorFactory.seed = 42


def detect_english(text: str):
    """Check whether the text 'looks like' English."""
    # Treat as English if langdetect detect text as en (even w/ small prob)
    try:
        for lang_result in detect_langs(text):
            if lang_result.lang == "en" and lang_result.prob > 0:
                return True
    except lang_detect_exception.LangDetectException:
        pass

    # Otherwise, treat as English if there is no non-English character
    return NON_ENGLISH_RE.search(text) is None


class SegmentType(enum.Enum):
    HEADING = enum.auto()   # Heading: HTML <h*> tags
    LISTITEM = enum.auto()  # List-item: HTML lists (unordered and ordered)
    TEXT = enum.auto()      # All other text segments


class DocumentSegment(NodeMixin):
    """Represents a text segment in the privacy policy"""

    def __init__(self, segment_id: int, segment_type: SegmentType, doc: Doc, parent=None):
        super().__init__()

        self.segment_id = segment_id
        self.segment_type = segment_type
        self.parent = parent

        self.context = [s := self]
        while (s := s.parent) is not None:
            self.context.append(s)

        self.set_doc(doc)

    def set_doc(self, doc: Doc):
        self.tokens = [str(t) for t in doc]
        self.spaces = [t.whitespace_ != "" for t in doc]

    @property
    def text(self):
        with io.StringIO() as fout:
            for token, space in zip(self.tokens, self.spaces):
                fout.write(token + (" " if space else ""))

            return fout.getvalue()

    def __repr__(self):
        return f"Segment #{self.segment_id}, type={self.segment_type.name}"


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
    """Container of privacy policy document"""

    @classmethod
    def initialize(cls, workdir, nlp: Language):
        obj = cls(flag=True)
        obj.workdir = Path(workdir)
        obj.token_relationship = nx.MultiDiGraph()

        with open(obj.workdir / "accessibility_tree.json", encoding="utf-8") as fin:
            accessibility_tree = json.load(fin)

        extractor = SegmentExtractor(accessibility_tree, nlp.tokenizer)
        obj.segments = extractor.extract()

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

        obj.all_docs = {}

        for doc in nlp.pipe(docs, batch_size=256):
            # See: https://github.com/explosion/spaCy/discussions/7486
            doc._.trf_data = None

            doc_id = doc.user_data["id"]
            obj.all_docs[doc_id] = doc

        return obj

    @classmethod
    def load(cls, workdir, nlp: Language):
        obj = cls(flag=True)
        obj.workdir = Path(workdir)

        with open(obj.workdir / "document.pickle", "rb") as fin:
            (obj.token_relationship, obj.segments, docbin_bytes) = pickle.load(fin)

        serialized_docs = DocBin().from_bytes(docbin_bytes)
        obj.all_docs = {}

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
        self.token_relationship: nx.MultiDiGraph
        self.segments: list[DocumentSegment]

    def print_tree(self):
        with io.StringIO() as fout:
            for seg in self.segments:
                if seg.depth == 0:
                    for pre, _, node in RenderTree(seg):
                        print(f"{pre}{node}, text={repr(node.text)}", file=fout)

            return fout.getvalue()

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

    def get_doc_with_context(self, segment: DocumentSegment) -> Doc:
        for i in range(len(segment.context) - 1, 0, -1):
            if (segment.segment_id, i) in self.all_docs:
                return self.all_docs[(segment.segment_id, i)]

        return self.all_docs[(segment.segment_id, 0)]

    def get_doc_without_context(self, segment: DocumentSegment) -> Doc:
        return self.all_docs[(segment.segment_id, 0)]

    def get_token_with_src(self, src: tuple) -> Token:
        segment = self.segments[src[0]]
        long_doc = self.get_doc_with_context(segment)
        rmap = long_doc.user_data["source_rmap"]
        return long_doc[rmap[src]]

    def link(self, token1, token2, relationship):
        src1 = token1._.src
        src2 = token2._.src

        if src1 is None or src2 is None:
            raise ValueError("Invalid token link")

        self.token_relationship.add_edge(src1, src2, key=relationship)

    def get_relations(self, token1, token2):
        src1 = token1._.src
        src2 = token2._.src

        if src1 is None or src2 is None:
            raise ValueError("Invalid tokens")

        yield from self.token_relationship.get_edge_data(src1, src2)

    def get_all_links(self, token: Token, direction="out"):
        doc = token.doc
        source_rmap = doc.user_data["source_rmap"]

        if token._.src is None:
            return

        match direction:
            case "out":
                edge_view = self.token_relationship.out_edges(token._.src, keys=True)
            case "in":
                edge_view = self.token_relationship.in_edges(token._.src, keys=True)
            case _:
                raise ValueError(f"Invalid direction: {direction}")

        for src, dst, relationship in edge_view:
            src_dst_tokens = []

            for token_src in src, dst:
                try:
                    token = doc[source_rmap[token_src]]
                except KeyError:
                    token = self.get_token_with_src(token_src)

                src_dst_tokens.append(token)

            yield src_dst_tokens[0], src_dst_tokens[1], relationship


IGNORED_ELEMENTS = frozenset({
    "img", "figure", "image map", "separator", "whitespace", "form", "radio",
    "list item marker", "insertion", "diagram", "dialog", "tab", "checkbox",
    "menu", "menubar", "menuitem", "parent menuitem", "toolbar", "tooltip",
    "internal frame", "listbox", "progressbar", "spinbutton", "switch",
    "details", "summary", "alert", "status", "mark", "note",
    "button", "buttonmenu", "slider", "textbox", "deletion",
    "combobox", "combobox list", "combobox option", "gridcell", "option",
    "application", "canvas", "caption", "toggle button", "password text",
    # TODO: are definition lists relevant?
    "definition", "definitionlist", "term",
    # FIXME: can tree role be lists?
    "tree", "treeitem", "treegrid",
})
SECTION_ELEMENTS = frozenset({
    "document", "article", "landmark", "section", "blockquote", "group",
    "tablist", "tabpanel", "region",
    # TODO: treat table elements as sections before we actually handle them
    "table", "row", "rowgroup",
})
TEXT_CONTAINER_ELEMENTS = frozenset({
    "paragraph", "text", "link", "statictext", "label", "text container", "text leaf", "superscript", "subscript",
    "cell", "columnheader", "rowheader", # TODO: table cells
})


class SegmentExtractor:
    """Process an accessibility tree into a list of DocumentSegment"""

    # NOTE:
    # heading: "heading"
    # list: "list", "listitem"
    # table: "table", "columnheader", "row", "cell"
    # definition list: "definitionlist", "term", "definition"

    @staticmethod
    def collect_text_from_children(node):
        if node["role"] in IGNORED_ELEMENTS:
            return

        if "children" in node:
            for child in node.get("children", []):
                yield from SegmentExtractor.collect_text_from_children(child)
        else:
            yield node["name"].strip()

    @staticmethod
    def extract_text(*node_list):
        # Concat text from children nodes
        all_text = []

        for node in node_list:
            all_text.extend(SegmentExtractor.collect_text_from_children(node))

        inner_text = " ".join(all_text)

        if not detect_english(inner_text):
            return ""

        # Strip non-ASCII text whenever possible
        inner_text = unidecode(inner_text)

        # FIXME: workaround common tokenizer errors
        inner_text = re.sub(r"\be-mails?\b", "email", inner_text, flags=re.I)
        inner_text = re.sub(r"\bwi-fi\b", "WiFi", inner_text, flags=re.I)
        inner_text = re.sub(r"\bgeo-location\b", "geolocation", inner_text, flags=re.I)
        inner_text = re.sub(r"\bid\b", "ID", inner_text)
        inner_text = re.sub(r"\b(\w+)\(s\)", r"\1s", inner_text)  # e.g. App(s) => "App(s" + ")"

        return inner_text

    @staticmethod
    def find_listitems(list_node):
        for child in list_node.get("children", []):
            if child["role"] == "listitem":
                yield child
            else:
                # An illformed list. Try to either:
                #   find any nested listitems;
                #   or, create a fake listitem child.
                found_nothing = True

                for item in SegmentExtractor.find_listitems(child):
                    found_nothing = False
                    yield item

                if found_nothing:
                    yield {"role": "listitem", "children": [child]}

    def __init__(self, tree, nlp_tokenizer):
        self.tree = tree
        self.tokenizer = nlp_tokenizer

        self.segments: list[DocumentSegment] = []
        self.headings: list[tuple[DocumentSegment, int]] = []

        self.current_html_path: list[int] = []
        self.parent_html_paths: list[list[int]] = []

    def new_segment(self, segment_type, text, parent):
        seg = DocumentSegment(len(self.segments), segment_type, self.tokenizer(text), parent)
        logging.info("New segment: %r, Parent: %r", seg, parent)

        self.segments.append(seg)
        self.parent_html_paths.append(self.current_html_path[:-1])

        return seg

    def iterate(self, node):
        if node["role"] in IGNORED_ELEMENTS:
            return
        elif node["role"] in SECTION_ELEMENTS:
            for idx, child in enumerate(node.get("children", [])):
                self.current_html_path.append(idx)
                self.iterate(child)
                self.current_html_path.pop()
        elif node["role"] == "heading":
            level = node["level"]

            while self.headings and self.headings[-1][1] >= level:
                self.headings.pop()

            parent = self.headings[-1][0] if self.headings else None
            heading = self.new_segment(SegmentType.HEADING, self.extract_text(node), parent)
            self.headings.append((heading, level))
        elif node["role"] == "list":
            if self.segments and self.parent_html_paths[-1] == self.current_html_path[:-1]:
                parent = self.segments[-1]
            else:
                parent = self.headings[-1][0] if self.headings else None

            for child in self.find_listitems(node):
                listitem = None
                text_element_queue = []

                for idx, grandchild in enumerate(chain(child.get("children", []), [None])):
                    if grandchild is None or grandchild["role"] in SECTION_ELEMENTS or grandchild["role"] == "list":
                        self.current_html_path.append(idx)

                        if current_text := self.extract_text(*text_element_queue):
                            # Lazy creation to avoid empty LISTITEM
                            listitem = listitem or self.new_segment(SegmentType.LISTITEM, "", parent)
                            self.new_segment(SegmentType.TEXT, current_text, listitem)
                            text_element_queue.clear()

                        if grandchild is not None:
                            self.iterate(grandchild)

                        self.current_html_path.pop()
                    else:
                        text_element_queue.append(grandchild)
        elif node["role"] in TEXT_CONTAINER_ELEMENTS:
            if text := self.extract_text(node):
                parent = self.headings[-1][0] if self.headings else None
                self.new_segment(SegmentType.TEXT, text, parent)
        else:
            raise ValueError(f"Invalid role: {node['role']}")

    def fix_non_html_lists(self):
        """Turns non-HTML text-only bullets into listitem segments

        Some poor-written webpages use text bullets (e.g. `a.`, `b.`) instead
        of HTML lists. Try to identify such lists and convert them from text
        segments into listitem segments.
        """

        need_renumber = False
        i = 1  # Intentionally skip the first one

        while i < len(self.segments):
            before_seg = self.segments[i - 1]

            if (self.segments[i].segment_type == SegmentType.TEXT and
                (bullet_matcher := BulletMatcher.init(self.segments[i].text)) and
                before_seg.segment_type in [SegmentType.TEXT, SegmentType.HEADING] and
                before_seg.text.endswith(":")):

                parent_seg = self.segments[i].parent
                j = i

                while (j < len(self.segments) and
                       self.segments[j].segment_type == SegmentType.TEXT and
                       bullet_matcher.match(self.segments[j].text) and
                       self.segments[j].parent == parent_seg):
                    j += 1

                if (num_segments := j - i) > 1:
                    logging.info("Convert text segments [%d, %d] to list items", i, j - 1)

                    for k in range(num_segments):
                        current_seg = self.segments[i + k * 2]

                        listitem_seg = DocumentSegment(0, SegmentType.LISTITEM, self.tokenizer(""), before_seg)
                        current_seg.parent = listitem_seg

                        new_text = bullet_matcher.trim_bullet(current_seg.text)
                        new_text = new_text or current_seg.text  # FIXME: avoid empty segment (only bullet point)
                        current_seg.set_doc(self.tokenizer(new_text))

                        self.segments.insert(i + k * 2, listitem_seg)

                    need_renumber = True
                    i += k * 2 + 2
                else:
                    i += 1
            else:
                i += 1

        if need_renumber:
            logging.info("Re-number segment IDs")

            for idx, seg in enumerate(self.segments):
                seg.segment_id = idx

    def extract(self):
        self.segments.clear()
        self.parent_html_paths.clear()
        self.headings.clear()

        self.iterate(self.tree)
        self.fix_non_html_lists()

        return self.segments


class BulletMatcher:
    @staticmethod
    def init(text):
        if m := re.match(r"^[*>#-]", text):
            return BulletMatcher(m.re, itertools.cycle(m[0]))
        elif m := re.match(r"^\W*\d+\W", text):
            tpl = m[0].replace("1", "{0}")
            return BulletMatcher(m.re, map(tpl.format, itertools.count(1)))
        elif m := re.match(r"^\W*[a-z]\W", text):
            tpl = m[0].replace("a", "{0}")
            return BulletMatcher(m.re, map(tpl.format, itertools.cycle(string.ascii_lowercase)))
        elif m := re.match(r"^\W*[A-Z]\W", text):
            tpl = m[0].replace("A", "{0}")
            return BulletMatcher(m.re, map(tpl.format, itertools.cycle(string.ascii_uppercase)))

        return None

    def __init__(self, re_pattern: re.Pattern, iterator: Iterable[str]):
        self.regex = re_pattern
        self.iterator = iterator

    def match(self, text: str):
        return text.startswith(next(self.iterator))

    def trim_bullet(self, text: str):
        return self.regex.sub("", text, count=1).strip()
