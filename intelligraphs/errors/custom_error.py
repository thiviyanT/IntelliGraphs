class DataError(Exception):
    """Custom exception for data-related errors."""
    def __init__(self, message):
        super().__init__(message)