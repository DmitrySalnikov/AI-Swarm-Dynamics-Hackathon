# Число состояний в кубическом ящике

- id: `ladder_rung_2/quantum_mechanics`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r2_box_states`, energy bound E_max in units of E1 (actual value p minus 0..9) = 430, сид 2000
- отбор: медиана 18316 ток., CV 0.20, верно 8/8

## Условие

```text
A particle is confined in a cubic box with impenetrable walls. Its energy levels are E = (n_x² + n_y² + n_z²)·E1, where n_x, n_y, n_z = 1, 2, 3, … are positive integers and E1 = h²/(8mL²). Every ordered triple (n_x, n_y, n_z) is a distinct quantum state (ignore spin); for example, (1, 1, 2), (1, 2, 1) and (2, 1, 1) are three different states.

How many quantum states have energy E ≤ 429·E1? Give the count as an integer.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`4181` — проверка: число, точное совпадение
