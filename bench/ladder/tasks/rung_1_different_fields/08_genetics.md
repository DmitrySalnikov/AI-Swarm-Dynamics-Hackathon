# Отбор в однолокусной модели

- id: `ladder_rung_1/genetics`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r1_gen_selection`, number of generations = 10, сид 1000
- отбор: медиана 19058 ток., CV 0.16, верно 5/5

## Условие

```text
A diploid population has one locus with two alleles, A and a. Genotype fitnesses (relative viabilities) are w(AA) = 0.70, w(Aa) = 0.80, w(aa) = 0.95. Generations are discrete and non-overlapping, mating is random (Hardy–Weinberg proportions among zygotes), there is no mutation, migration or drift. If p is the frequency of A in one generation, the frequency in the next generation is

  p' = (p²·w(AA) + p·q·w(Aa)) / (p²·w(AA) + 2·p·q·w(Aa) + q²·w(aa)),  where q = 1 − p.

The initial frequency is p₀ = 0.88. Rounding rule: after computing each generation, round p' to 4 decimal places (round half up) and use this rounded value for the next generation.

What is the frequency of A after 10 generations (p_10)? Give it rounded to 4 decimal places.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`0.6379` — проверка: число (rel_tol 0.001)
