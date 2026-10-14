# Следы степеней матрицы

- id: `ladder_rung_3/power_traces`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r3_traces`, matrix size n (powers 1..4) = 8, сид 3000
- отбор: медиана 18995 ток., CV 0.13, верно 5/5

## Условие

```text
Consider the following 8×8 integer matrix A.

A =
[  0  -2   2  -1   0  -1   2   2 ]
[  2   0   0   0   2   0   0   1 ]
[ -2   0  -1   1   1   0   0   0 ]
[  0   0  -2   0   0   0  -1   0 ]
[ -1  -1   0   2   0   0   1   0 ]
[  0   0   2   0   0  -2   0   2 ]
[  0   2  -1   0   0  -2   0   0 ]
[  1   0   0  -1   0   0   0   2 ]

Compute the traces of its powers: tr(A), tr(A²), tr(A³), tr(A⁴). Give these 4 numbers in this order as one comma-separated list of integers.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`-1, -11, 59, 121` — проверка: список через запятую, точное совпадение
