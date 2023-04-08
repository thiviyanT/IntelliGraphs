from intelligraphs import Intelligraphs

# Create an instance of Intelligraphs with 50 random triples
intelligraph = Intelligraphs(num_triples=50)

# Get the list of triples
triples = intelligraph.triples

# Get the natural language sentences for the triples
sentences = intelligraph.to_natural_language()

# Print the sentences
for sentence in sentences:
    print(sentence)
