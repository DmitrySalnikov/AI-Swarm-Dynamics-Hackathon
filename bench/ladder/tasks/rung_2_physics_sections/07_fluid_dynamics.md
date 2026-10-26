# Давление в трубопроводе

- id: `ladder_rung_2/fluid_dynamics`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r2_pipeline_bernoulli`, number of sections N = 6, сид 2001
- отбор: медиана 19964 ток., CV 0.25, верно 8/8 (принята как максимально близкая: CV выше 0,25)

## Условие

```text
Water (density 1000 kg/m³, incompressible) flows steadily at a volume flow rate of 7 L/s through a pipeline of 6 sections connected one after another:
section 1: cross-section 123 cm², centreline elevation 11.2 m; loss coefficient of the connection to section 2: K = 0.2
section 2: cross-section 80 cm², centreline elevation 9.5 m; loss coefficient of the connection to section 3: K = 1.3
section 3: cross-section 17 cm², centreline elevation 11.8 m; loss coefficient of the connection to section 4: K = 0.4
section 4: cross-section 44 cm², centreline elevation 18.2 m; loss coefficient of the connection to section 5: K = 0.8
section 5: cross-section 44 cm², centreline elevation 23.0 m; loss coefficient of the connection to section 6: K = 1.2
section 6: cross-section 129 cm², centreline elevation 19.6 m

In every section the speed is uniform over the cross-section (v = Q/A). Between section i and section i + 1 the flow loses pressure Δp_loss = K·ρ·v_i²/2, where v_i is the speed in the upstream section i; otherwise the Bernoulli equation holds: p + ρv²/2 + ρgz is conserved. Use g = 9.8 m/s². The absolute pressure in section 1 is 375 kPa.

Find the absolute pressure in section 6, in kPa, with 4 significant figures.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`286.243` — проверка: число (rel_tol 0.002)
