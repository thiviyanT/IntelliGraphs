import pytest

def test_import_intelligraphs():
    """
    Test importing the 'intelligraphs' package and check for the '__version__' attribute.
    """
    try:
        import intelligraphs
        assert hasattr(intelligraphs, '__version__'), "intelligraphs does not have a __version__ attribute."
        print(f"intelligraphs version: {intelligraphs.__version__}")
    except ImportError:
        pytest.fail("Failed to import the 'intelligraphs' package.")


def test_import_baseline_models():
    """
    Test importing baseline models from the 'intelligraphs.baselines' module.
    """
    models_to_import = [
        "uniform_baseline_model",
        "probabilistic_kge_model"
    ]
    module_name = "intelligraphs.baselines"

    for model_name in models_to_import:
        try:
            module = __import__(module_name, fromlist=[model_name])
            getattr(module, model_name)
        except (ImportError, AttributeError):
            pytest.fail(f"Failed to import '{model_name}' from {module_name}.")


def test_import_data_generators():
    """
    Test importing data generators from the 'intelligraphs.generators.synthetic' module.
    """
    generator_to_import = {
        "intelligraphs.generators.synthetic": ["SynPathsGenerator", "SynTIPRGenerator", "SynTypesGenerator"],
        # "intelligraphs.generators.wikidata": ["WDMovies", "WDArticles"]
    }

    for module_name, class_names in generator_to_import.items():
        for class_name in class_names:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
            except (ImportError, AttributeError):
                pytest.fail(f"Failed to import '{class_name}' from {module_name}.")


def test_import_verifiers():
    """
    Test importing verifiers from the 'intelligraphs.verifier' and related modules.
    """
    verifiers_to_import = {
        "intelligraphs.verifier": ["ConstraintVerifier"],
        "intelligraphs.verifier.synthetic": ["SynPathsVerifier", "SynTypesVerifier", "SynTIPRVerifier"],
        "intelligraphs.verifier.wikidata": ["WDMoviesVerifier", "WDArticlesVerifier"]
    }

    for module_name, class_names in verifiers_to_import.items():
        for class_name in class_names:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
            except (ImportError, AttributeError):
                pytest.fail(f"Failed to import '{class_name}' from {module_name}.")


def test_import_data_loaders():
    """
    Test importing the 'IntelliGraphsDataLoader' from 'intelligraphs.data_loaders'.
    """
    try:
        from intelligraphs.data_loaders import IntelliGraphsDataLoader
    except ImportError:
        pytest.fail("Failed to import 'IntelliGraphsDataLoader' from intelligraphs.")


def test_import_evaluators():
    """
    Test importing evaluator functions from the 'intelligraphs.evaluators' module.
    """
    evaluators_to_import = [
        "check_semantics",
        "is_graph_empty",
        "is_graph_in_training_data",
        "validate_graph",
        "compile_results"
    ]
    module_name = "intelligraphs.evaluators"

    for evaluator_name in evaluators_to_import:
        try:
            module = __import__(module_name, fromlist=[evaluator_name])
            getattr(module, evaluator_name)
        except (ImportError, AttributeError):
            pytest.fail(f"Failed to import '{evaluator_name}' from {module_name}.")


def test_import_domains():
    """
    Test importing classes and attributes from various domain modules within 'intelligraphs.domains'.
    """
    imports_to_test = {
        "intelligraphs.domains.SynPaths.entities": ["dutch_cities"],
        "intelligraphs.domains.SynPaths": ["relations"],
        "intelligraphs.domains.SynTIPR.entities": ["names", "roles", "years"],
        "intelligraphs.domains.SynTIPR": ["relations"],
        "intelligraphs.domains.SynTypes.entities": ["cities", "countries", "languages"],
        "intelligraphs.domains.SynTypes": ["relations"],
        "intelligraphs.domains.WDArticles.entities": ["author_positions", "authors", "papers", "ordinals", "subjects"],
        "intelligraphs.domains.WDArticles": ["relations"],
        "intelligraphs.domains.WDMovies.entities": ["actors", "directors", "genres"],
        "intelligraphs.domains.WDMovies": ["relations"],
    }

    for module_name, attributes in imports_to_test.items():
        try:
            module = __import__(module_name, fromlist=attributes)
            if "*" in attributes:
                # Import everything (for cases with '*')
                continue
            for attr in attributes:
                getattr(module, attr)
        except (ImportError, AttributeError):
            pytest.fail(f"Failed to import from '{module_name}' with attributes {attributes}.")
