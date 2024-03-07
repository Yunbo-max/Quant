# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2023-12-31 09:14:40
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-03-07 10:11:11
from scipy.special import comb

def calculate_probability(n, m):
    probability = 0.0
    for k in range(n+1):
        inner_sum = 0.0
        for j in range(k + 1,n+m+1):
            inner_sum += comb(n + m, j) * (0.5 ** j) * (0.5 ** (n + m - j))
        probability += comb(n, k) * (0.5 ** k) * (0.5 ** (n -k)) * inner_sum
    return probability

# Example usage:
n_values = [1, 2, 3, 1000]
m_values = [1, 2]

for m in m_values:
    for n in n_values:
        p = calculate_probability(n, m)
        print(f"P(n={n}, m={m}) = {p:.5f}")
