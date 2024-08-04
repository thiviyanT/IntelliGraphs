import torch
from torch import nn
from .scoring_functions import TransE, DistMult, Complex


class KGEModel(nn.Module):
    """
    Simple link predictor based on traditional scoring functions

    Outputs raw (linear) scores for the given triples.
    """

    def __init__(self, n, r, embedding=512, decoder='distmult', edropout=None, rdropout=None, init=0.85,
                 biases=False, init_method='uniform', reciprocal=False):

        super().__init__()

        self.n, self.r = n, r
        self.e = embedding
        self.reciprocal = reciprocal

        if decoder == 'distmult':
            self.decoder = DistMult(embedding, n, r, init_method, reciprocal, init, edropout, rdropout)
        elif decoder == 'transe':
            self.decoder = TransE(embedding, n, r, init_method, reciprocal, init, edropout, rdropout)
        elif decoder == 'complex':
            self.decoder = Complex(embedding, n, r, init_method, reciprocal, init, edropout, rdropout)
        else:
            raise Exception()

        self.biases = biases
        if biases:
            self.gbias = nn.Parameter(torch.zeros((1,)))
            self.sbias = nn.Parameter(torch.zeros((n,)))
            self.obias = nn.Parameter(torch.zeros((n,)))
            self.pbias = nn.Parameter(torch.zeros((r,)))

            if reciprocal:
                self.pbias_bw = nn.Parameter(torch.zeros((r,)))

    def forward(self, s, p, o, recip=None):
        """
        Takes a batch of triples in s, p, o indices, and computes their scores.

        If s, p and o have more than one dimension, and the same shape, the resulting score
        tensor has that same shape.

        If s, p and o have more than one dimension and mismatching shape, they are broadcast together
        and the score tensor has the broadcast shape. If broadcasting fails, the method fails. In order to trigger the
        correct optimizations, it's best to ensure that all tensors have the same dimensions.

        :param s:
        :param p:
        :param o:
        :param recip: prediction mode, if this is a reciprocal model. 'head' for head prediction, 'tail' for tail
            prediction, 'eval' for the average of both (i.e. for final scoring).
        :return:
        """

        assert recip in [None, 'head', 'tail', 'eval']
        assert self.reciprocal or (recip is None), 'Predictor must be set to model reciprocal relations for recip to be set'
        if self.reciprocal and recip is None:
            recip = 'eval'

        scores = 0

        if recip is None or recip == 'tail':
            modes = [True]  # forward only
        elif recip == 'head':
            modes = [False]  # backward only
        elif recip == 'eval':
            modes = [True, False]
        else:
            raise Exception(f"{recip}")
            pass

        for forward in modes:

            si, pi, oi = (s, p, o) if forward else (o, p, s)

            scores = scores + self.decoder(si, pi, oi)
            # -- We let the decoder handle the broadcasting

            if self.biases:
                pb = self.pbias if forward else self.pbias_bw

                scores = scores + (self.sbias[si] + pb[pi] + self.obias[oi] + self.gbias)

        if self.reciprocal:
            scores = scores / len(modes)

        return scores

    def penalty(self, rweight, p, which):

        if which == 'entities':
            params = [self.decoder.entities]
        elif which == 'relations':
            if self.reciprocal:
                params = [self.decoder.relations, self.decoder.relations_backward]
            else:
                params = [self.decoder.relations]
        else:
            raise Exception()

        if p % 2 == 1:
            params = [p.abs() for p in params]

        return (rweight / p) * sum([(p ** p).sum() for p in params])
