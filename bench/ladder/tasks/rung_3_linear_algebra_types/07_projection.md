# Ортогональная проекция на подпространство

- id: `ladder_rung_3/projection`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r3_projection`, number of spanning vectors k (in R^(k+2)) = 6, сид 3000
- отбор: медиана 16692 ток., CV 0.11, верно 5/5

## Условие

```text
In R^8 with the standard dot product, let W be the subspace spanned by the 6 linearly independent vectors

w1 = (-1, -1, 1, 1, 0, 1, -1, 1)
w2 = (0, 0, 0, 1, -1, 0, 0, 1)
w3 = (-2, -1, 2, 0, 2, 1, -2, -1)
w4 = (0, 0, 0, 0, 1, 1, -1, 1)
w5 = (-1, 0, 1, 0, 0, 0, -1, -1)
w6 = (3, 1, -2, 1, -1, 0, 2, 3)

and let

b = (1, -5, 4, 3, -2, 1, -2, -1)

Compute the orthogonal projection of b onto W. Give its 8 coordinates as one comma-separated list of exact numbers (integers or exact fractions a/b where needed).

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`-14/15, -46/15, 71/15, 34/15, -2, 11/5, -1/15, -4/15` — проверка: список через запятую, точное совпадение
