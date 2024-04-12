from logging import getLogger
from methodtools import lru_cache
import itertools

import numpy as np
import networkx as nx


def centerOfMass(members, rpos):
    # logger = getLogger()
    dtotal = np.zeros(3)
    origin = rpos[list(members)[0]]
    for member in members:
        d = rpos[member] - origin
        d -= np.floor(d + 0.5)
        dtotal += d
    com = rpos[members[0]] + dtotal / len(members)
    com -= np.floor(com)
    return com


# Modified from CountRings class in gtihub/vitroid/countrngs
def cycles_iter(graph, maxsize, pos=None):
    """
    A generator of cycles in a graph.
    The graph must not be directed.
    Specify the positions of the vertices in a orthogonal cell in the fractional coordinate if you want to avoid the spanning cycles.
    """
    # shortes_pathlen is a stateless function, so the cache is useful to avoid
    # re-calculations.
    @lru_cache(maxsize=None)
    def _shortest_pathlen(graph, pair):
        return len(nx.shortest_path(graph, *pair)) - 1

    def _shortcuts(graph, members):
        n = len(members)
        for i in range(0, n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                if d > _shortest_pathlen(
                    graph, frozenset(
                        (members[i], members[j]))):
                    return True
        return False

    def _findring(graph, members, maxsize):
        # print members, "maxsize:", maxsize
        if len(members) > maxsize:
            return (maxsize, [])
        s = set(members)
        last = members[-1]
        results = []
        for adj in graph[last]:
            if adj in s:
                if adj == members[0]:
                    # Ring is closed.
                    # It is the best and unique answer.
                    if not _shortcuts(graph, members):
                        return (len(members), [members])
                else:
                    # Shortcut ring
                    pass
            else:
                (newmax, newres) = _findring(graph, members + [adj], maxsize)
                if newmax < maxsize:
                    maxsize = newmax
                    results = newres
                elif newmax == maxsize:
                    results += newres
        return (maxsize, results)

    def _is_spanning(cycle):
        "Return True if the cycle spans the periodic cell."
        total = np.zeros_like(pos[cycle[0]])
        N = len(cycle)
        for i, j in zip(cycle[-1:N-1], cycle):
            d = pos[i] - pos[j]
            d -= np.floor(d + 0.5)
            total += d
        return np.any(np.absolute(total) > 1e-5)

    logger = getLogger()
    rings = set()
    for x in graph:
        neis = sorted(graph[x])
        for y, z in itertools.combinations(neis, 2):
            triplet = [y, x, z]
            (_max, results) = _findring(graph, triplet, maxsize)
            for i in results:
                # Make i immutable for the key.
                j = frozenset(i)
                # and original list as the value.
                if j not in rings:
                    # logger.debug("({0}) {1}".format(len(i),i))
                    if pos is None or not _is_spanning(i):
                        yield tuple(i)
                        rings.add(j)


# def graph_overlap(g1, g2):
#     """
#     Return the common part of two graphs.
#
#     They are adjacent if two subgraphs share at least one vertex.
#     """
#     # common vertices
#     common = g1.nodes() and g2.nodes()
#     g = nx.Graph()
#     for v in common:
#         g.add_node(v)
#     if len(common) > 0:
#         # common edges
#         for e in g1.edges():
#             if g2.has_edge(*e):
#                 g.add_edge(*e)
#     return g


def test():
    g = nx.Graph()
    # a lattice graph of 4x4x4
    X, Y, Z = np.meshgrid(np.arange(4.0), np.arange(4.0), np.arange(4.0))
    X = X.reshape(64)
    Y = Y.reshape(64)
    Z = Z.reshape(64)
    coord = np.array([X, Y, Z]).T
    # fractional coordinate
    coord /= 4
    for a in range(64):
        for b in range(a):
            d = coord[b] - coord[a]
            # periodic boundary condition
            d -= np.floor(d + 0.5)
            # if adjacent
            if d @ d < 0.3**2:
                g.add_edge(a, b)
    # PBC-compliant
    A = set([cycle for cycle in cycles_iter(g, 4, pos=coord)])
    print(f"Number of cycles (PBC compliant): {len(A)}")
    print(A)

    # not PBC-compliant
    B = set([cycle for cycle in cycles_iter(g, 4)])
    print(f"Number of cycles (crude)        : {len(B)}")
    print(B)

    # difference
    C = B - A
    print("Cycles that span the cell:")
    print(C)

    # g1 = nx.Graph([(1,2),(2,3),(3,4)])
    # g2 = nx.Graph([(1,4)])
    # print(graph_overlap(g1,g2))


if __name__ == "__main__":
    test()
