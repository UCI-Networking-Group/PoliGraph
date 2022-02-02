#!/usr/bin/env python3
"""Process accessibility tree into DocumentSegment objects"""

import argparse
import json
from pathlib import Path


class DocumentSegment:
    def __init__(self, segment_type, link, text):
        self.type = segment_type
        self.link = link
        self.text = text

    def __repr__(self):
        if self.type == "listitem":
            return f"({self.type}) FOLLOWING {repr(self.link)}"
        else:
            return f"({self.type}) {repr(self.text)}"

    def get_text_with_context(self):
        text_buffer = []
        current_segment = self
        while current_segment is not None:
            if current_segment.type == "listitem":
                text_buffer.append(" ")
            elif current_segment.type == "text":
                text_buffer.append(current_segment.text)
            elif current_segment.type == "heading":
                text_buffer.append("\n\n")
                text_buffer.append(current_segment.text)

            current_segment = current_segment.link

        return "".join(reversed(text_buffer)).strip()


def process_accessibility_tree(tree):
    heading_stack = [(-1, None)]
    segments = []

    def extract_text(node):
        if node["role"] in ["img", "list item marker"]:
            return

        if "children" in node:
            for child in node.get("children", []):
                yield from extract_text(child)
        else:
            yield node["name"].strip()

    def iterate(node):
        if node["role"] == "table":
            # TODO: parse tables
            return

        if node["role"] in ["document", "landmark", "section"]:
            for child in node.get("children", []):
                iterate(child)
        elif node["role"] == "heading":
            text = " ".join(extract_text(node))
            level = node["level"]

            while heading_stack[-1][0] >= level:
                heading_stack.pop()

            link = heading_stack[-1][1]
            heading_segment = DocumentSegment("heading", link, text)
            heading_stack.append((level, heading_segment))
            segments.append(heading_segment)
        elif node["role"] == "list":
            link = segments[-1]

            for child in node.get("children", []):
                if child["role"] != "listitem":
                    raise ValueError("Invalid child element of a list")

                listitem_segment = DocumentSegment("listitem", link, "")
                segments.append(listitem_segment)
                text_buffer = []

                for grandchild in child.get("children", []):
                    if grandchild["role"] == "list":
                        if len(text_buffer) > 0:
                            seg_in_list = DocumentSegment("text", listitem_segment, " ".join(text_buffer))
                            segments.append(seg_in_list)
                            text_buffer.clear()

                        iterate(grandchild)
                    else:
                        text_buffer.extend(extract_text(grandchild))

                if len(text_buffer) > 0:
                    seg_in_list = DocumentSegment("text", listitem_segment, " ".join(text_buffer))
                    segments.append(seg_in_list)
        elif node["role"] in ["paragraph", "link"]:
            link = heading_stack[-1][1]

            text = " ".join(extract_text(node))
            if text:
                segments.append(DocumentSegment("text", link, text))
        else:
            raise ValueError(f"Invalid role: {node['role']}")

    iterate(tree)
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", help="working directory")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    with open(workdir / "accessibility_tree.json") as fin:
        accessibility_tree = json.load(fin)

    segments = process_accessibility_tree(accessibility_tree)
    for seg in segments:
        print(seg)
        print(seg.get_text_with_context())


if __name__ == "__main__":
    main()
