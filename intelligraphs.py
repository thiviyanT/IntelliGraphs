import random
from typing import List, Tuple


class Intelligraphs:
    def __init__(self, num_triples: int = 100):
        """
        Initialize the Intelligraphs instance.

        Args:
            num_triples (int): The number of random triples to generate. Default is 100.
        """
        self.num_triples = num_triples
        self.triples = self.generate_random_triples()

    def generate_random_triples(self) -> List[Tuple[str, str, str]]:
        """
        Generate random triples to create a synthetic Knowledge Graph.

        Returns:
            List[Tuple[str, str, str]]: A list of random triples.
        """
        subjects = ["Alice", "Bob", "Charlie", "David", "Eve"]
        predicates = ["likes", "dislikes", "knows", "loves", "hates"]
        objects = ["pizza", "ice cream", "sushi", "football", "music"]

        triples = []
        for _ in range(self.num_triples):
            subject = random.choice(subjects)
            predicate = random.choice(predicates)
            obj = random.choice(objects)
            triples.append((subject, predicate, obj))

        return triples

    def to_natural_language(self) -> List[str]:
        """
        Generate a list of natural language sentences representing the triples.

        Returns:
            List[str]: A list of natural language sentences.
        """
        sentences = []
        for triple in self.triples:
            subject, predicate, obj = triple
            sentence = f"{subject} {predicate} {obj}."
            sentences.append(sentence)

        return sentences
