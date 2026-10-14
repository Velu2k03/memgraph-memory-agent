import argparse

from memgraph.cli import cmd_cypher


def _args(query):
    return argparse.Namespace(query=query)


def test_cypher_rejects_write_keywords():
    assert cmd_cypher(_args("MATCH (n) CREATE (m)")) == 2
    assert cmd_cypher(_args("MATCH (n) CALL { WITH n MERGE (m) } RETURN n")) == 2
    assert cmd_cypher(_args("DELETE (n)")) == 2


def test_cypher_rejects_non_read_prefix():
    assert cmd_cypher(_args("CREATE (n:Entity)")) == 2
