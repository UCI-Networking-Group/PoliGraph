from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# This is where the candidate labels are defined.
PURPOSE_LABEL_MAPPING = {
    "service": "services",
    "functionality": "services",
    "transaction": "services",
    "maintenance": "services",
    "operation": "services",
    "security": "security",
    "authorization": "security",
    "authentication": "security",
    "legal": "legal",
    "liability": "legal",
    "acquisition": "legal",
    "analytics": "analytics",
    "research": "analytics",
    "advertising": "advertising",
    "marketing": "advertising",
}
CLASSIFICATION_LABELS = list(PURPOSE_LABEL_MAPPING.keys())
THRESHOLD_1 = 0.95
THRESHOLD_2 = 0.60

class PurposeClassifier:
    def __init__(self, is_multi_label=True):
        # Initialize the NLP model and classifier.
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
        model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        #tokenizer = AutoTokenizer.from_pretrained('typeform/distilbert-base-uncased-mnli')
        #model = AutoModelForSequenceClassification.from_pretrained('typeform/distilbert-base-uncased-mnli')
        self.classifier = pipeline(task='zero-shot-classification', tokenizer=tokenizer, model=model, device=0)

        self.is_multi_label = is_multi_label

    def __call__(self, text):
        def get_main_label(result_dict):
            labels = set()

            for label, score in zip(result_dict["labels"], result_dict["scores"]):
                relabel = PURPOSE_LABEL_MAPPING[label]

                if score > THRESHOLD_1:
                    labels.add(relabel)
                elif len(labels) == 0 and score > THRESHOLD_2:
                    labels.add(relabel)
                    break
                else:
                    break

            return labels

        results = self.classifier(sequences=text, candidate_labels=CLASSIFICATION_LABELS,
                                  multi_label=self.is_multi_label,
                                  num_workers=0)  # to prevent HuggingFace from spawning a lot of processes

        if isinstance(results, dict):
            return get_main_label(results)
        else:
            return [get_main_label(r) for r in results]
