from intelligraphs.verifier.synthetic import SynPathsVerifier, SynTIPRVerifier, SynTypesVerifier
from intelligraphs.verifier.wikidata import WDArticlesVerifier, WDMoviesVerifier
from intelligraphs.data_loaders import load_data_as_tensor
from intelligraphs.evaluators import post_process_data


def test_syn_paths():
    print('Validating syn-paths dataset')
    print('Preparing the data... ', end="", flush=True)
    data = load_data_as_tensor('syn-paths', padding=True)
    print("(done)")
    train, val, test, (e2i, i2e), (r2i, i2r), _, _ = data

    # short_circuit=False checking further rules once a violation is found.
    verifier = SynPathsVerifier(short_circuit=False)

    # Preprocess the data
    train_processed = post_process_data(train, i2e, i2r)
    val_processed = post_process_data(val, i2e, i2r)
    test_processed = post_process_data(test, i2e, i2r)

    # Verify the train, validation, and test datasets using the method in ConstraintVerifier
    verifier.verify_dataset(train_processed, "training")
    verifier.verify_dataset(val_processed, "validation")
    verifier.verify_dataset(test_processed, "test")
    print('\n\n')


def test_syn_tipr():
    print('Validating syn-tipr dataset')
    print('Preparing the data... ', end="", flush=True)
    data = load_data_as_tensor('syn-tipr', padding=True)
    print("(done)")
    train, val, test, (e2i, i2e), (r2i, i2r), _, _ = data

    # short_circuit=True stops checking further rules once a violation is found.
    verifier = SynTIPRVerifier(short_circuit=False)

    # Preprocess the data
    train_processed = post_process_data(train, i2e, i2r)
    val_processed = post_process_data(val, i2e, i2r)
    test_processed = post_process_data(test, i2e, i2r)

    # Verify the train, validation, and test datasets using the method in ConstraintVerifier
    verifier.verify_dataset(train_processed, "training")
    verifier.verify_dataset(val_processed, "validation")
    verifier.verify_dataset(test_processed, "test")
    print('\n\n')


def test_syn_types():
    print('Validating syn-types dataset')
    print('Preparing the data... ', end="", flush=True)
    data = load_data_as_tensor('syn-types', padding=True)
    print("(done)")
    train, val, test, (e2i, i2e), (r2i, i2r), _, _ = data

    # short_circuit=False checking further rules once a violation is found.
    verifier = SynTypesVerifier(short_circuit=False)

    # Preprocess the data
    train_processed = post_process_data(train, i2e, i2r)
    val_processed = post_process_data(val, i2e, i2r)
    test_processed = post_process_data(test, i2e, i2r)

    # Verify the train, validation, and test datasets using the method in ConstraintVerifier
    verifier.verify_dataset(train_processed, "training")
    verifier.verify_dataset(val_processed, "validation")
    verifier.verify_dataset(test_processed, "test")
    print('\n\n')


def test_wd_movies():
    print('Validating wd-movies dataset')
    print('Preparing the data... ', end="", flush=True)
    data = load_data_as_tensor('wd-movies', padding=True)
    train, val, test, (e2i, i2e), (r2i, i2r), _, _ = data
    print("(done)")
    # short_circuit=False checking further rules once a violation is found.
    verifier = WDMoviesVerifier(short_circuit=False)

    # Preprocess the data
    train_processed = post_process_data(train, i2e, i2r)
    val_processed = post_process_data(val, i2e, i2r)
    test_processed = post_process_data(test, i2e, i2r)

    # Verify the train, validation, and test datasets using the method in ConstraintVerifier
    verifier.verify_dataset(train_processed, "training")
    verifier.verify_dataset(val_processed, "validation")
    verifier.verify_dataset(test_processed, "test")
    print('\n\n')


def test_wd_articles():
    print('Validating wd-articles dataset')
    print('Preparing the data... ', end="", flush=True)
    data = load_data_as_tensor('wd-articles', padding=True)
    train, val, test, (e2i, i2e), (r2i, i2r), _, _ = data
    print("(done)")
    # short_circuit=False checking further rules once a violation is found.
    verifier = WDArticlesVerifier(short_circuit=False)

    # Preprocess the data
    train_processed = post_process_data(train, i2e, i2r)
    val_processed = post_process_data(val, i2e, i2r)
    test_processed = post_process_data(test, i2e, i2r)

    # Verify the train, validation, and test datasets using the method in ConstraintVerifier
    verifier.verify_dataset(train_processed, "training")
    verifier.verify_dataset(val_processed, "validation")
    verifier.verify_dataset(test_processed, "test")
    print('\n\n')


if __name__ == "__main__":
    test_syn_paths()
    test_syn_tipr()
    test_syn_types()
    test_wd_movies()
    test_wd_articles()
    print("All datasets passed the semantic check.")
