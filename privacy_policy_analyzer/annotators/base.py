import logging
from abc import ABC, abstractmethod

from spacy import Language

from ..document import PolicyDocument


class BaseAnnotator(ABC):
    def __init__(self, nlp: Language):
        self.nlp = nlp
        self.vocab = nlp.vocab
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def annotate(self, document: PolicyDocument):
        pass
