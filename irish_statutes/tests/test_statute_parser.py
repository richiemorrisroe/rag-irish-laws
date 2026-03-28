from pathlib import Path

from indexer.parse_statute import StatuteParser


RAW_HTML = Path(__file__).parent.parent / "raw_html"

ACT_38 = RAW_HTML / "2014" / "act_38.html" ##companies act, 2014

def test_statute_parser_exists():
    s = StatuteParser(path = ACT_38)
    assert s is not None

def test_statute_parser_has_a_statute_node():    
    s = StatuteParser(path = ACT_38)
    assert s.statute_nodes is not None
