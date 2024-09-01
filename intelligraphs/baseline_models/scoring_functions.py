import torch
from torch import nn
from torch.nn.functional import normalize
from abc import abstractmethod
from intelligraphs.baseline_models.utils import initialize


class Decoder(nn.Module):
    def __init__(self, e, n, r, init_method, reciprocal, init, edropout, rdropout):
        super().__init__()

        self.e = e
        self.entities = nn.Parameter(torch.FloatTensor(n, self.e))
        initialize(self.entities, init_method)
        self.relations = nn.Parameter(torch.FloatTensor(r, self.e))
        initialize(self.relations, init_method)

        if reciprocal:
            self.relations_backward = nn.Parameter(torch.FloatTensor(r, self.e).uniform_(-init, init))
            initialize(self.relations_backward, init_method)

        self.edo = None if edropout is None else nn.Dropout(edropout)
        self.rdo = None if rdropout is None else nn.Dropout(rdropout)

    def s_dim(self):
        return self.e

    def p_dim(self):
        return self.e

    def o_dim(self):
        return self.e

    @abstractmethod
    def forward(self, si, pi, oi):
        pass


class TransE(Decoder):
    def __init__(self, e, n, r, init_method, reciprocal, init, edropout, rdropout):
        super().__init__(e, n, r, init_method, reciprocal, init, edropout, rdropout)

    def forward(self, si, pi, oi):
        """
        Implements the transe score function.
        """

        # Apply dropout
        nodes = self.entities if self.edo is None else self.edo(self.entities)
        relations = self.relations if self.rdo is None else self.rdo(self.relations)

        s, p, o = nodes[si, :], relations[pi, :], nodes[oi, :]
        return (s + p - o).sum(dim=-1)


class DistMult(Decoder):
    def __init__(self, e, n, r, init_method, reciprocal, init, edropout, rdropout):
        super().__init__(e, n, r, init_method, reciprocal, init, edropout, rdropout)

    def forward(self, si, pi, oi):
        """
        Implements the distmult score function.
        """

        # Apply dropout
        nodes = self.entities if self.edo is None else self.edo(self.entities)
        relations = self.relations if self.rdo is None else self.rdo(self.relations)

        s = normalize(nodes[si, :], p=2, dim=1)
        p = relations[pi, :]
        o = normalize(nodes[oi, :], p=2, dim=1)

        return (s * p * o).sum(dim=-1)


class Complex(Decoder):
    def __init__(self, e, n, r, init_method, reciprocal, init, edropout, rdropout):
        super().__init__(e, n, r, init_method, reciprocal, init, edropout, rdropout)

        self.imagery_entities = nn.Parameter(torch.FloatTensor(n, self.e))
        initialize(self.imagery_entities, init_method)
        self.imagery_relations = nn.Parameter(torch.FloatTensor(r, self.e))
        initialize(self.imagery_relations, init_method)

    def forward(self, si, pi, oi):
        """
        Implements the complex score function.
        """

        # Apply dropout
        nodes = self.entities if self.edo is None else self.edo(self.entities)
        relations = self.relations if self.rdo is None else self.rdo(self.relations)

        nodes_i = self.imagery_entities if self.edo is None else self.edo(self.imagery_entities)
        relations_i = self.imagery_relations if self.rdo is None else self.rdo(self.imagery_relations)

        # Real and imaginary parts of the entities and relations
        re_s, im_s = nodes[si, :], nodes_i[si, :]
        re_r, im_p = relations[pi, :], relations_i[pi, :]
        re_r, im_o = nodes[oi, :], nodes_i[oi, :]

        return (re_s * (re_r * re_r + im_p * im_o) + im_s * (re_r * im_o - im_p * re_r)).sum(dim=-1)

