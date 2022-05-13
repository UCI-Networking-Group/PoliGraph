import argparse

from privacy_policy_analyzer import named_entity_recognition as ner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ner", help="NER model directory")
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()

    nlp = ner.setup_models(args.ner)
    nlp.to_disk(args.output)


if __name__ == "__main__":
    main()
