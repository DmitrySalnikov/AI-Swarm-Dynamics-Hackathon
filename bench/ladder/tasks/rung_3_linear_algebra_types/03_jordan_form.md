# Жорданова форма

- id: `ladder_rung_3/jordan_form`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r3_jordan`, matrix size n = 7, сид 3000
- отбор: медиана 19309 ток., CV 0.18, верно 4/5

## Условие

```text
Consider the following 7×7 integer matrix A. Its only eigenvalue is 1.

A =
[  3   3   1  -5   0  -3   4 ]
[  1   2   1   1   0   1  -1 ]
[ -4  -6  -1   7  -1   4  -5 ]
[  2   1   3   1  -2   1   1 ]
[ -2  -4   0   3  -1   2  -1 ]
[  1   2   0  -3   1  -1   2 ]
[  3   3   3  -3  -1  -1   4 ]

Determine the Jordan normal form of A. Describe every Jordan block by the pair (eigenvalue, block size). Order the blocks by eigenvalue in increasing order and, for equal eigenvalues, by block size in decreasing order, and write all pairs one after another as one comma-separated list of integers λ, size, λ, size, … (for example, blocks J_2(−1), J_1(−1), J_3(4) give −1, 2, −1, 1, 4, 3).

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`1, 4, 1, 2, 1, 1` — проверка: список через запятую, точное совпадение
