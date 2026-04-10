from pathlib import Path

from indexer.parse_statute import StatuteNode, StatuteParser


RAW_HTML = Path(__file__).parent.parent / "raw_html"

ACT_38 = RAW_HTML / "2014" / "act_38.html" ##companies act, 2014

def test_statute_parser_exists():
    s = StatuteParser(path = ACT_38)
    assert s is not None

def test_statute_parser_has_a_statute_node():    
    s = StatuteParser(path = ACT_38)
    assert s.statute_nodes is not None


def test_statute_parser_can_read_file():
    s = StatuteParser(path = ACT_38)
    assert s.read() is not None

def test_statute_parser_can_parse_html():
    s = StatuteParser(path = ACT_38)
    s.read()
    assert s.parse() is not None

def test_statute_parser_can_store_statute_node():
    s = StatuteParser(path = ACT_38)
    s.read()
    s.parse()
    assert isinstance(s.statute_nodes, StatuteNode)


def test_statute_parser_can_get_ranks():
    s = StatuteParser(path = ACT_38)
    s.read()
    assert isinstance(s.get_ranks(), dict)

def test_statute_parser_has_a_stack():
    s = StatuteParser(path = ACT_38)
    s.read()
    assert s.stack is not None


