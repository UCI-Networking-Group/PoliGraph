#!/usr/bin/env python3
"""Runs spaCy NLP pipeline to annotate privacy policies and save the results.

Subsequent scripts will access NLP annotations again and again. For efficiency,
we just run the pipeline once with this script.

Check `PolicyDocument.initialize` for details.
"""

import argparse
import logging
import os

import spacy
import torch

from poligrapher.document import PolicyDocument
from poligrapher.utils import setup_nlp_pipeline


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("--nlp", default="", help="NLP model directory")
    parser.add_argument("--debug", action="store_true", help="Show NER results")
    parser.add_argument("--gpu-memory-threshold", default=0.9, type=float,
                        help="Max GPU usage to trigger manual cache cleaning")
    args = parser.parse_args()

    use_gpu = spacy.prefer_gpu()
    nlp = setup_nlp_pipeline(args.nlp)

    for d in args.workdirs:
        logging.info("Processing %s ...", d)

        document = PolicyDocument.initialize(d, nlp=nlp)
        document.save()

        if args.debug:
            with open(os.path.join(d, "document.txt"), "w", encoding="utf-8") as fout:
                fout.write(document.print_tree())

        if use_gpu:
            current_device = torch.cuda.current_device()
            gmem_total = torch.cuda.get_device_properties(current_device).total_memory
            gmem_reserved = torch.cuda.memory_reserved(current_device)

            if gmem_reserved / gmem_total > args.gpu_memory_threshold:
                logging.warning("Empty GPU cache...")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
