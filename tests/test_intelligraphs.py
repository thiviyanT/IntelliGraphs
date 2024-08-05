import pytest
from intelligraphs.generators import SyntheticGraphGenerator


def test_import_intelligraphs():
    try:
        import intelligraphs
        print(intelligraphs.__version__)
        assert True  # If import is successful, pass the test
    except ImportError:
        assert False, "Failed to import the 'intelligraphs' package."


def test_generate_graph():
    generator = SyntheticGraphGenerator()
    result = generator.generate_graphs(nodes=10, edges=15)
    assert len(result.nodes) == 10
    assert len(result.edges) == 15


def test_invalid_parameters():
    generator = SyntheticGraphGenerator()
    with pytest.raises(ValueError):
        generator.generate_graphs(nodes=-1, edges=5)
