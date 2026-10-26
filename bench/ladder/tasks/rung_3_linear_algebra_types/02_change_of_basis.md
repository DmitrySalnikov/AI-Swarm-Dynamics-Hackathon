# Матрица оператора в новом базисе

- id: `ladder_rung_3/change_of_basis`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r3_change_basis`, matrix size n = 6, сид 3000
- отбор: медиана 15266 ток., CV 0.20, верно 8/8

## Условие

```text
A linear operator on R^6 has the matrix A in the standard basis. The columns of the matrix B form a new basis (det B = ±1).

A =
[  3   0   0   3  -2  -3 ]
[ -1   0   0  -4  -2   0 ]
[  3   0   0  -1   4   0 ]
[  0   4   0   0   2   0 ]
[  0   4   4   0  -2  -3 ]
[  0   0   1   0   0   1 ]

B =
[  1   0   1   1   1   1 ]
[  1   1   2   2   0   0 ]
[ -1  -1  -3  -2  -1  -1 ]
[  1   0   2   2   1   2 ]
[  1   1   3   3   1   1 ]
[ -1  -1  -1  -1   0   0 ]

Find the matrix of the operator in the new basis, C = B⁻¹·A·B. Give all 36 entries of C row by row (first row left to right, then the second row, and so on) as one comma-separated list of integers.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`-1, -2, -2, -6, 3, 2, 12, 8, 25, 27, 6, 11, -16, -9, -25, -28, -8, -11, 7, 5, 6, 10, 0, -1, 9, -2, 0, 4, 3, 10, 8, 9, 27, 26, 6, 7` — проверка: список через запятую, точное совпадение
