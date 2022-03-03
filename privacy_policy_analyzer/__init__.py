from spacy.tokens import Token, Span

from privacy_policy_analyzer.utils import token_to_ent, token_to_ent_id, chunk_to_conjuncts

Token.set_extension("ent", getter=token_to_ent)
Token.set_extension("ent_id", getter=token_to_ent_id)
Span.set_extension("conjunct_chunks", getter=chunk_to_conjuncts)
