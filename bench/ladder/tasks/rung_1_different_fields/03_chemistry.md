# Состав смесей солей по потере массы

- id: `ladder_rung_1/chemistry`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r1_chem_mixture`, number of samples = 5, сид 1000
- отбор: медиана 17782 ток., CV 0.23, верно 8/8

## Условие

```text
Each sample below is a mixture of exactly two compounds. Every sample is heated strongly until its mass no longer changes. On heating the compounds behave as follows (water vapour and carbon dioxide escape completely; the solid products shown are stable and remain in the residue; nothing else happens):
  NaHCO3: 2 NaHCO3 → Na2CO3 + H2O + CO2
  MgCO3: MgCO3 → MgO + CO2
  CaCO3: CaCO3 → CaO + CO2
  ZnCO3: ZnCO3 → ZnO + CO2
  CuSO4·5H2O: CuSO4·5H2O → CuSO4 + 5 H2O
  Na2CO3·10H2O: Na2CO3·10H2O → Na2CO3 + 10 H2O
  NaCl: NaCl does not change
  K2CO3: K2CO3 does not change

Atomic masses: H = 1.008, C = 12.011, O = 15.999, Na = 22.990, Mg = 24.305, S = 32.06, Cl = 35.45, K = 39.098, Ca = 40.078, Cu = 63.546, Zn = 65.38.

Samples (initial mass of the mixture; mass of the solid residue after heating):
1. 19.02 g of a mixture of ZnCO3 and Na2CO3·10H2O; mass of the residue 8.11 g.
2. 13.09 g of a mixture of Na2CO3·10H2O and CuSO4·5H2O; mass of the residue 5.76 g.
3. 5.84 g of a mixture of K2CO3 and CaCO3; mass of the residue 4.04 g.
4. 26.48 g of a mixture of K2CO3 and MgCO3; mass of the residue 24.13 g.
5. 22.60 g of a mixture of NaHCO3 and NaCl; mass of the residue 19.18 g.

For each sample, find the mass percentage of the FIRST-named compound in the original mixture. Use the atomic masses above and keep at least 5 significant figures in intermediate results. Give the 5 percentages rounded to 1 decimal place, separated by commas, in the order of the samples.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`20.1,74.1,29.9,83.0,41.0` — проверка: список через запятую (rel_tol 0.003)
