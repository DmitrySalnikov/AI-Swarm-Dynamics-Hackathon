# Теплопередача через многослойную стенку

- id: `ladder_rung_2/heat_transfer`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r2_composite_wall`, number of layers N = 8, сид 2000
- отбор: медиана 14268 ток., CV 0.25, верно 8/8 (принята как максимально близкая: CV выше 0,25)

## Условие

```text
A flat composite wall consists of 8 layers in perfect thermal contact, listed from the inside to the outside:
layer 1: glass wool, thickness 25 mm, thermal conductivity 0.04 W/(m·K)
layer 2: plaster, thickness 280 mm, thermal conductivity 0.5 W/(m·K)
layer 3: gypsum board, thickness 120 mm, thermal conductivity 0.17 W/(m·K)
layer 4: stone, thickness 140 mm, thermal conductivity 2.3 W/(m·K)
layer 5: cork, thickness 30 mm, thermal conductivity 0.043 W/(m·K)
layer 6: concrete, thickness 285 mm, thermal conductivity 1.4 W/(m·K)
layer 7: glass wool, thickness 30 mm, thermal conductivity 0.04 W/(m·K)
layer 8: gypsum board, thickness 70 mm, thermal conductivity 0.17 W/(m·K)

The inside air is at 21 °C with a convective heat-transfer coefficient of 9 W/(m²·K) at the inner surface; the outside air is at -6 °C with a coefficient of 20 W/(m²·K) at the outer surface. Heat flow is steady and one-dimensional (no radiation, no edge effects).

Find (1) the heat flux through the wall in W/m² (from inside to outside) and (2) the temperature in °C at the interface between layer 3 and layer 4. Give both with 4 significant figures, in the form `q, T`.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`6.46571,8.05568` — проверка: список через запятую (rel_tol 0.002)
