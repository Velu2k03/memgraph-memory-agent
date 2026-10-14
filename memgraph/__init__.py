"""MemGraph: an AI assistant that actually remembers.

Every conversation becomes a knowledge graph stored in FalkorDB:
facts are extracted as (subject, relation, object) triples, linked to the
memories they came from, and recalled later through graph search.
"""

__version__ = "0.1.0"
