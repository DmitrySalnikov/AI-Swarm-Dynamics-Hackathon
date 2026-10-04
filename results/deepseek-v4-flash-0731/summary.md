# DeepSeek-V4-Flash-0731, reasoning_effort=high, one run per question

set_1 and set_2 — max_tokens 100000, set_3 and set_4 — max_tokens 5000.

## set_1

| Domain | Question | Reference answer | Model answer | Correct | Reasoning tokens | Time, s |
|---|---|---|---|---|---|---|
| computer_science | Breaking Diffie–Hellman, p=1009 | `760` | `760` | ✅ | 14064 | 189 |
| mathematics | Probability: 12 letters into pairs | `821` | `821` | ✅ | 7029 | 100 |
| finance | Retirement annuity | `1898.27` | `$1,898.27` | ✅ | 17939 | 177 |
| linguistics | Verb forms of Guazacapán Xinka | `ɨnnetakʼa, ɨŋɡɨrʼɨ, ɨmpʼuhurʼu, ɨnherʼo` | `ɨnnetakʼa, ɨŋɡɨrʼɨ, ɨmpʼuhurʼu, ɨnherʼo` | ✅ | 26040 | 375 |
| physics | Mass of an incandescent lamp filament | `0.00168` | `0.0106 kg` | ❌ | 14386 | 159 |
| biology | GC content of a sequence | `49` | `49` | ✅ | 9240 | 159 |
| chemistry | Number of stereoisomers | `16` | `16` | ✅ | 3211 | 84 |
| classics | Latin hexameter scansion | `DDSSDS` | `D D S S D S` | ✅ | 24065 | 509 |
| economics | Welfare at market equilibrium | `22.572` | `22.767` | ❌ | 63821 | 1178 |
| music | Just intonation: "Hänschen klein" | `62720/243` | `62720/243` | ✅ | 2995 | 43 |

Solved: **8 / 10**

## set_2

| Domain | Question | Reference answer | Model answer | Correct | Reasoning tokens | Time, s |
|---|---|---|---|---|---|---|
| computer_science | Output of a Scheme program with call/cc | `1121314` | `1121314` | ✅ | 13105 | 177 |
| mathematics | Quadrilateral formed by circumcenters (9√15) | `34.857` | `34.857` | ✅ | 4560 | 36 |
| finance | PE ratio | `28.75` | `28.75` | ✅ | 4831 | 53 |
| linguistics | Matching Zuni words | `I, G, H, C, J, D, E, B, A, F` | `H, B, F, E, J, D, C, G, I, A` | ❌ | 21573 | 621 |
| physics | Olympiad physics problem | `2216` | `2216.49` | ✅ | 23873 | 460 |
| biology | GC content of a sequence | `49` | `49` | ✅ | 5188 | 36 |
| chemistry | Synthesis from cyclohexanone: H types | `6` | `6` | ✅ | 36007 | 739 |
| classics | Scansion of a line of Plautus | `LSS SL SL LL SSL SS` | `LSS SL SL LL SSL SL` | ❌ | 47195 | 778 |
| economics | Job search and unemployment benefit | `0.218` | `0.113` | ❌ | 37208 | 681 |
| music | Just intonation: second melody | `216513/800` | `216513/800` | ✅ | 5060 | 64 |

Solved: **7 / 10**

## set_3

| Domain | Question | Reference answer | Model answer | Correct | Reasoning tokens | Time, s |
|---|---|---|---|---|---|---|
| computer_science | Diffie–Hellman, p=227 | `99` | `99` | ✅ | 2217 | 27 |
| mathematics | Numbers using digits 1–8 divisible by 22 | `279` | `279` | ✅ | 1947 | 49 |
| finance | Project NPV | `189.26` | `$189.26` | ✅ | 2283 | 38 |
| linguistics | Numerals of an invented language | `47, 83, 121` | `47, 83, 121` | ✅ | 1755 | 22 |
| physics | Parachutist with quadratic drag | `345.0` | `344.8745` | ✅ | 4658 | 69 |
| biology | GC content of 150 nucleotides | `51` | `51` | ✅ | 1470 | 15 |
| chemistry | NaHCO3 and MgCO3 mixture from mass loss | `71.1` | `71.1` | ✅ | 1892 | 17 |
| classics | Roman calendar dates | `27.02, 14.10, 02.05, 16.12, 08.07, 22.05` | `27.02, 14.10, 02.05, 16.12, 08.07, 22.05` | ✅ | 2822 | 43 |
| economics | Certainty equivalent of an offer with a bonus | `108610` | `108610` | ✅ | 4792 | 63 |
| music | Just intonation: 13-note melody | `31360/81` | `31360/81` | ✅ | 2359 | 50 |

Solved: **10 / 10**

## set_4

| Domain | Question | Reference answer | Model answer | Correct | Reasoning tokens | Time, s |
|---|---|---|---|---|---|---|
| geometry | Distance between the circumcenter and the orthocenter | `2.2361` | `2.2361` | ✅ | 899 | 9 |
| linear_algebra | Determinant of a 5×5 integer matrix | `-970` | `-970` | ✅ | 3615 | 46 |
| calculus | Improper integral ∫₀^∞ x²e⁻ˣ sin x dx | `0.5000` | `0.5` | ✅ | 961 | 22 |
| differential_equations | y″ + 2y′ + 5y = 10 cos x, find y(π) | `-2.0864` | `-2.0864` | ✅ | 1007 | 16 |
| series | Double series | `0.28125` | `0.28125` | ✅ | 2468 | 49 |
| number_theory | Divisors of 9! ending in 1 | `103` | `103` | ✅ | 1086 | 20 |
| combinatorics | Coloring the segments of a 2×2 grid | `82` | `82` | ✅ | 3112 | 60 |
| probability | Random subset of the divisors of 2025 | `237` | `237` | ✅ | 981 | 22 |
| algebra | Product of logarithms | `106` | `106` | ✅ | 1111 | 20 |
| complex_numbers | System with moduli of complex numbers | `77` | `77` | ✅ | 1664 | 37 |

Solved: **10 / 10**
