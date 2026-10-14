# Ток в ветви лестничной цепи

- id: `ladder_rung_2/dc_circuits_mesh`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r2_ladder_mesh`, number of loops N = 4, сид 2000
- отбор: медиана 15840 ток., CV 0.25, верно 7/8 (принята как максимально близкая: CV выше 0,25)

## Условие

```text
A planar ladder network has 4 loops. Its top row consists of nodes A0, A1, …, A4; at the bottom there is a single common wire G of zero resistance. Neighbouring top nodes are joined by resistors, and every top node Ai is joined to G by a vertical branch i. The elements are:
- branch 0, from A0 down to G: resistor 14 Ω (no battery)
- between A0 and A1: resistor 13 Ω
- branch 1, from A1 down to G: resistor 1 Ω (no battery)
- between A1 and A2: resistor 2 Ω
- branch 2, from A2 down to G: resistor 4 Ω in series with an ideal battery of EMF 8 V whose positive terminal faces G
- between A2 and A3: resistor 13 Ω
- branch 3, from A3 down to G: resistor 20 Ω (no battery)
- between A3 and A4: resistor 5 Ω
- branch 4, from A4 down to G: resistor 1 Ω in series with an ideal battery of EMF 4 V whose positive terminal faces G

All batteries are ideal (no internal resistance). Find the current in branch 3, counted positive when it flows from A3 through the branch down to G (negative if it flows from G up to A3). Give the value in amperes with 4 significant figures, including its sign.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`-0.157767` — проверка: число (rel_tol 0.002)
