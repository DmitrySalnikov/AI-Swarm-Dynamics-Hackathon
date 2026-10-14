# Длины волн водородоподобных ионов

- id: `ladder_rung_2/atomic_physics`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r2_hydrogenlike_lines`, number of transitions K = 35, сид 2000
- отбор: медиана 18857 ток., CV 0.17, верно 5/5

## Условие

```text
For each of the following 35 transitions in a hydrogen-like ion (one electron, nuclear charge Z), find the wavelength of the emitted photon. Use the Rydberg formula 1/λ = R·Z²·(1/n_f² − 1/n_i²) with R = 1.0973731568·10^7 m⁻¹ (infinitely heavy nucleus; ignore reduced-mass, fine-structure and all other corrections).

1. Z = 4, transition n = 7 → n = 3
2. Z = 6, transition n = 6 → n = 3
3. Z = 4, transition n = 7 → n = 3
4. Z = 6, transition n = 7 → n = 4
5. Z = 6, transition n = 6 → n = 1
6. Z = 3, transition n = 6 → n = 3
7. Z = 1, transition n = 3 → n = 1
8. Z = 6, transition n = 6 → n = 5
9. Z = 8, transition n = 11 → n = 5
10. Z = 3, transition n = 7 → n = 2
11. Z = 7, transition n = 8 → n = 3
12. Z = 2, transition n = 5 → n = 4
13. Z = 1, transition n = 6 → n = 4
14. Z = 6, transition n = 8 → n = 2
15. Z = 7, transition n = 7 → n = 2
16. Z = 8, transition n = 7 → n = 2
17. Z = 4, transition n = 6 → n = 4
18. Z = 1, transition n = 10 → n = 4
19. Z = 5, transition n = 4 → n = 1
20. Z = 3, transition n = 10 → n = 4
21. Z = 1, transition n = 6 → n = 4
22. Z = 6, transition n = 7 → n = 4
23. Z = 8, transition n = 9 → n = 5
24. Z = 2, transition n = 7 → n = 4
25. Z = 7, transition n = 8 → n = 5
26. Z = 2, transition n = 4 → n = 1
27. Z = 4, transition n = 5 → n = 3
28. Z = 1, transition n = 10 → n = 5
29. Z = 6, transition n = 10 → n = 5
30. Z = 3, transition n = 3 → n = 1
31. Z = 7, transition n = 8 → n = 2
32. Z = 4, transition n = 6 → n = 4
33. Z = 6, transition n = 4 → n = 3
34. Z = 7, transition n = 7 → n = 2
35. Z = 6, transition n = 8 → n = 3

Give the 35 wavelengths in nanometres, each with 4 significant figures, in the order listed, as plain decimal numbers (no units, no exponent notation, no thousands separators) separated by commas.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`62.792,30.3756,62.792,60.1375,2.60362,121.502,102.518,207.106,44.8663,44.1008,19.4764,1012.52,2624.45,10.8002,8.10015,6.20168,164.028,1735.75,3.88807,192.861,2624.45,60.1375,51.4876,541.237,76.2966,24.3005,80.0918,3037.56,84.3766,11.3908,7.93484,164.028,52.0724,8.10015,26.5096` — проверка: список через запятую (rel_tol 0.002)
