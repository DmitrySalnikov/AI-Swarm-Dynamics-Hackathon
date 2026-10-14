# Базис ядра матрицы

- id: `ladder_rung_3/null_space`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r3_nullspace`, number of columns n = 10, сид 3000
- отбор: медиана 15716 ток., CV 0.07, верно 5/5

## Условие

```text
Consider the following 9×10 integer matrix A.

A =
[  -1   -1   -1   -1   -1    0   -1   -6   -1    1 ]
[   1    2    2    5    0    3    0   -3    2    0 ]
[   0    1    2    5   -2    4    0  -12    0    2 ]
[   1    2    3    6    0    2    2    0    1    1 ]
[   0    1    0    3    0    2   -1   -3    1   -1 ]
[  -1    0    1    4   -2    2    1   -9   -1    1 ]
[   1    1    1    1    2   -2    3   15   -1    0 ]
[  -1    0   -1    2   -1    2   -2   -9    1    0 ]
[   1    0   -1   -4    2   -2   -1    9    2   -2 ]

Find the canonical basis of the null space {x : A x = 0}: take the free variables of the reduced row echelon form of A; for each free variable x_f build the solution with x_f = 1 and all other free variables 0. List these basis vectors in order of increasing free-variable index f, and write each vector as its n entries x_1, …, x_n. Give all vectors one after another as one comma-separated list of integers (k·n numbers for a k-dimensional null space).

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`3, -3, -1, 1, 0, 0, 0, 0, 0, 0, -1, -2, 1, 0, 2, 1, 0, 0, 0, 0, -3, 0, 3, 0, -3, 0, -3, 1, 0, 0` — проверка: список через запятую, точное совпадение
