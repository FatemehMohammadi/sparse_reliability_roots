#!/usr/bin/env python3
"""Compute the reliability roots of random triangle complexes sampled at p = c/n."""
import argparse
import csv
import sys
from bisect import bisect_right
from fractions import Fraction
from itertools import combinations
from math import comb, expm1

import numpy as np


def triangle_from_rank(rank, pairs, triples):
    """Decode rank = C(k,3) + C(j,2) + i for 0 <= i < j < k."""
    k = bisect_right(triples, rank) - 1
    remainder = rank - triples[k]
    j = bisect_right(pairs, remainder, 0, k) - 1
    return remainder - pairs[j], j, k


def sample_faces(n, c, rng):
    total = comb(n, 3)
    m = int(rng.binomial(total, c / n))
    # Conditional on m, every m-subset of the potential triangles is equiprobable.
    ids = rng.choice(total, size=m, replace=False, shuffle=False)
    pairs = [comb(i, 2) for i in range(n)]
    triples = [comb(i, 3) for i in range(n)]
    vertices, edges = set(), set()
    for idx in ids:
        i, j, k = triangle_from_rank(int(idx), pairs, triples)
        vertices.update((i, j, k))
        edges.update(((i, j), (i, k), (j, k)))
    return len(vertices), len(edges), m


def h_vector(v, e, t):
    return (1, v - 3, e - 2 * v + 3, t - e + v - 1)


def check_formulas():
    n = 8
    pairs, triples = ([comb(i, d) for i in range(n)] for d in (2, 3))
    decoded = {triangle_from_rank(j, pairs, triples) for j in range(comb(n, 3))}
    assert decoded == set(combinations(range(n), 3))
    q = Fraction(2, 7)
    counts = (1, 5, 6, 2)  # two triangles meeting in one vertex
    h = h_vector(*counts[1:])
    assert sum(a * q ** k for k, a in enumerate(h)) == sum(
        f * q ** k * (1 - q) ** (3 - k) for k, f in enumerate(counts)
    )


def check_shadow_moments(n, d, p):
    """Compare the facet-minus-ridge moments to enumeration over all facet sets."""
    facets = list(combinations(range(n), d))
    T, L, m = len(facets), comb(n, d - 1), n - d + 1
    shadows = [set(combinations(facet, d - 1)) for facet in facets]
    probabilities = [p**f * (1 - p)**(T - f) for f in range(T + 1)]
    first = [Fraction(0), Fraction(0)]
    second = [Fraction(0), Fraction(0)]

    for mask in range(1 << T):
        f = mask.bit_count()
        ridges = set()
        for i, shadow in enumerate(shadows):
            if (mask >> i) & 1:
                ridges.update(shadow)
        for i, k in enumerate((1, 2)):
            value = f - k * len(ridges)
            first[i] += probabilities[f] * value
            second[i] += probabilities[f] * value**2

    z = (1 - p)**m  # probability that a given ridge is uncovered
    var_f = T * p * (1 - p)
    var_r = L * (z * (1 - z) + (d - 1) * m * p * (1 - p)**(2 * m - 1))
    cov_fr = L * m * p * z
    for i, k in enumerate((1, 2)):
        assert first[i] == T * p - k * L * (1 - z)
        assert second[i] - first[i]**2 == var_f + k**2 * var_r - 2 * k * cov_fr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', nargs='+', type=int, default=[45, 75])
    parser.add_argument('--c', nargs='+', type=float, default=[1., 4., 8.])
    parser.add_argument('--seed', type=int, default=1905)
    parser.add_argument('--draws', type=int, default=1)
    args = parser.parse_args()
    if min(args.n) < 3 or args.draws < 1 or args.seed < 0:
        parser.error('Require n >= 3, at least one draw, and a nonnegative seed.')
    if any(not 0 < c <= min(args.n) for c in args.c):
        parser.error('Require 0 < c <= n for every parameter pair.')
    check_formulas()
    check_shadow_moments(4, 2, Fraction(2, 5))
    check_shadow_moments(5, 3, Fraction(2, 5))
    rng = np.random.Generator(np.random.PCG64(args.seed))
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(('n', 'c', 'draw', 'vertices', 'edges', 'triangles',
                     'h0', 'h1', 'h2', 'h3', 'outer_real', 'outer_imag', 'limit'))
    for n in args.n:
        for c in args.c:
            a = -expm1(-c)
            prediction = a / (a - c / 3) if a != c / 3 else float('inf')
            for draw in range(1, args.draws + 1):
                counts = sample_faces(n, c, rng)
                h = h_vector(*counts)
                roots = np.roots(np.trim_zeros(h[::-1], 'f'))
                outer = max(roots, key=abs, default=complex(float('nan'), 0))
                writer.writerow((n, f'{c:g}', draw, *counts, *h,
                                 f'{outer.real:.6f}', f'{outer.imag:.6f}', f'{prediction:.6f}'))


if __name__ == '__main__':
    main()
