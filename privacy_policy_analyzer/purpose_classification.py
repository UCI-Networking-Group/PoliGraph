import logging

from setfit import SetFitModel

PURPOSE_LABELS = [
    "advertising",
    "analytics",
    "legal",
    "security",
    "services",
]

logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.WARNING)


class PurposeClassifier:
    def __init__(self, path: str):
        self.model = SetFitModel.from_pretrained(path)

    def __call__(self, text: list[str]) -> list[list[str]]:
        results = self.model(text)

        for i in range(len(text)):
            yield [PURPOSE_LABELS[j] for j in results[i, :].nonzero()]
