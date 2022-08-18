from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# This is where the candidate labels are defined.
PURPOSE_LABELS = [
    'acquisition',
    'advertising',
    'analytics',
    'services',
    'marketing',
    'legal',
    'personalization',
    'security'
]

class PurposeClassifier:
    def __init__(self, is_multi_label=True, threshold=0.5, multilabel_threshold=0.9):
        # Initialize the NLP model and classifier.
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
        model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        #tokenizer = AutoTokenizer.from_pretrained('typeform/distilbert-base-uncased-mnli')
        #model = AutoModelForSequenceClassification.from_pretrained('typeform/distilbert-base-uncased-mnli')
        self.classifier = pipeline(task='zero-shot-classification', tokenizer=tokenizer, model=model, device=0)

        self.is_multi_label = is_multi_label
        self.threshold = threshold
        self.multilabel_threshold = multilabel_threshold

    def __call__(self, text):
        def get_main_label(result_dict):
            labels = []

            if result_dict['scores'][0] > self.threshold:
                labels.append(result_dict['labels'][0])

            for score, label in zip(result_dict['scores'][1:], result_dict['labels'][1:]):
                if score > self.multilabel_threshold:
                    labels.append(label)
                else:
                    break

            return labels

        results = self.classifier(sequences=text, candidate_labels=PURPOSE_LABELS, multi_label=self.is_multi_label,
                                  num_workers=0)  # to prevent HuggingFace from spawning a lot of processes

        if isinstance(results, dict):
            return get_main_label(results)
        else:
            return [get_main_label(r) for r in results]