# Решение системы линейных уравнений

- id: `ladder_rung_3/linear_system`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r3_solve`, matrix size n = 10, сид 3000
- отбор: медиана 16674 ток., CV 0.14, верно 5/5

## Условие

```text
Consider the following 10×10 integer matrix A and vector b.

A =
[  -1   -2    1   -2    0   -2   -2    1    2    2 ]
[   0   -1   -1   -1    0    0   -2    2    2    2 ]
[   0   -2   -1   -2    2   -1   -6    4    5    2 ]
[  -2   -4    1   -5   -2   -2   -1    2    2    6 ]
[  -2   -2    5   -2    0   -5   -3   -3    2   -3 ]
[  -1   -3   -1   -4    0    1    2    2   -1    8 ]
[   2    4    0    5    8    0   -2    1   -2   -7 ]
[   0   -2    0   -2    2   -3  -10    5    9   -1 ]
[   2    3   -4    4   -2    2   -3    4    3   -2 ]
[   2    2   -4    1    2    5    2    5   -1   -1 ]

b (as a column) =
[  5 ]
[  6 ]
[  8 ]
[ -9 ]
[ -9 ]
[ -3 ]
[ -7 ]
[  2 ]
[ -1 ]
[  2 ]

Solve the linear system A x = b (it has a unique solution). Give x_1, …, x_n as one comma-separated list of exact numbers (integers or exact fractions a/b where needed).

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`3, -47, -21, 22, 9/2, -8, 13, -3, 6, -10` — проверка: список через запятую, точное совпадение
