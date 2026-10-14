# Степень матрицы

- id: `ladder_rung_3/matrix_power`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r3_power`, matrix size n (power 4) = 7, сид 3000
- отбор: медиана 14271 ток., CV 0.09, верно 4/5

## Условие

```text
Consider the following 7×7 integer matrix A.

A =
[  0   2   0   0   0   0   1 ]
[  0   0   0   0   0   2  -2 ]
[  0   1   2  -1   0  -2  -1 ]
[  1   2  -2  -1   1   0   0 ]
[  1  -2  -2   0   0   0   0 ]
[ -2   0   0   2   0   0  -1 ]
[  1   2   0   1  -1  -2  -1 ]

Compute the matrix power A^4 = A·A·A·A. Give all 49 entries of A^4 row by row (first row left to right, then the second row, and so on) as one comma-separated list of integers.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`-24, -34, -8, 30, -2, -8, -15, 22, -12, -32, -16, 8, -4, 38, 23, 20, 60, -50, -2, -58, -10, 21, -8, -64, 5, 11, 18, -14, 13, 28, -32, -12, 12, 58, 2, 4, 22, -32, 18, 2, -32, 21, -15, 2, -8, -21, 29, -14, -1` — проверка: список через запятую, точное совпадение
