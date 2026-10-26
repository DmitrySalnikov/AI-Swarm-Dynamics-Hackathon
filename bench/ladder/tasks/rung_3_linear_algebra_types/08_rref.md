# Приведённый ступенчатый вид

- id: `ladder_rung_3/rref`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r3_rref`, number of rows m (columns m+2) = 8, сид 3000
- отбор: медиана 18918 ток., CV 0.10, верно 5/5

## Условие

```text
Consider the following 8×10 integer matrix A.

A =
[   1    1    1    1    1    2    0   -1    1   -3 ]
[  -1    0    0    0   -1    0    1    2    0    8 ]
[   0    1    2    0   -4    4    2    0    0    4 ]
[   1    2    3    0   -6    6    1    0    1    1 ]
[   1    0    1    0    0    2    2   -5    0   -9 ]
[   1    2    3    0   -6    6    0    2   -1    4 ]
[   1    0   -1    0    2   -2   -3   -1    1  -10 ]
[  -1   -2   -3    0    6   -6   -2    1   -2   -1 ]

Compute the reduced row echelon form R of A (leading entries equal to 1, zeros above and below every leading 1, zero rows at the bottom). Give all 80 entries of R row by row (first row left to right, then the second row, and so on, including zero rows) as one comma-separated list of exact numbers (integers or exact fractions a/b where needed).

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 2, 0, 0, 0, -1, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0` — проверка: список через запятую, точное совпадение
