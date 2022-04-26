from spacy.tokens import Token, Span

from privacy_policy_analyzer import utils

Token.set_extension("src", getter=utils.token_to_source)
Token.set_extension("ent", getter=utils.token_to_ent)
Token.set_extension("ent_id", getter=utils.token_to_ent_id)
Token.set_extension("ent_type", getter=utils.token_ent_type)
Span.set_extension("conjunct_chunks", getter=utils.chunk_to_conjuncts)
Span.set_extension("ent_type", getter=utils.span_ent_type)
