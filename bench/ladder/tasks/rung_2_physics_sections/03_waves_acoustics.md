# Эффект Доплера

- id: `ladder_rung_2/waves_acoustics`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r2_doppler_cases`, number of cases K = 47, сид 2000
- отбор: медиана 19607 ток., CV 0.07, верно 5/5

## Условие

```text
Sound travels in still air at 343 m/s. In each of the following 47 independent situations all motion is along the straight line joining the source, the observer and (if present) the reflector, all speeds are constant, and any reflector re-emits the sound it receives as a moving source.
1. an observer moves at 76 m/s away from a stationary source emitting 929 Hz; frequency heard by the observer.
2. a source emitting 1190 Hz moves at 37 m/s away from a stationary observer; frequency heard by the observer.
3. a stationary source emitting 296 Hz sends sound to a large flat reflector that moves at 27 m/s toward the source; frequency of the reflected sound received back at the source.
4. a stationary source emitting 1021 Hz sends sound to a large flat reflector that moves at 40 m/s toward the source; frequency of the reflected sound received back at the source.
5. a source emitting 578 Hz moves at 34 m/s toward a stationary observer; frequency heard by the observer.
6. a source emitting 805 Hz moves at 49 m/s toward a stationary observer; frequency heard by the observer.
7. a source emitting 1630 Hz moves at 52 m/s away from the observer, and the observer moves at 67 m/s away from the source; frequency heard by the observer.
8. a source emitting 268 Hz moves at 59 m/s toward the observer, and the observer moves at 53 m/s away from the source; frequency heard by the observer.
9. a stationary source emitting 1257 Hz sends sound to a large flat reflector that moves at 77 m/s toward the source; frequency of the reflected sound received back at the source.
10. a source emitting 215 Hz moves at 26 m/s away from the observer, and the observer moves at 5 m/s away from the source; frequency heard by the observer.
11. an observer moves at 51 m/s toward a stationary source emitting 654 Hz; frequency heard by the observer.
12. an observer moves at 14 m/s away from a stationary source emitting 1426 Hz; frequency heard by the observer.
13. a source emitting 1169 Hz moves at 80 m/s away from a stationary observer; frequency heard by the observer.
14. a stationary source emitting 1081 Hz sends sound to a large flat reflector that moves at 12 m/s away from the source; frequency of the reflected sound received back at the source.
15. an observer moves at 50 m/s toward a stationary source emitting 1600 Hz; frequency heard by the observer.
16. a source emitting 1009 Hz moves at 81 m/s toward a stationary observer; frequency heard by the observer.
17. a source emitting 1993 Hz moves at 61 m/s toward a stationary observer; frequency heard by the observer.
18. a source emitting 1084 Hz moves at 50 m/s away from the observer, and the observer moves at 14 m/s toward the source; frequency heard by the observer.
19. an observer moves at 78 m/s away from a stationary source emitting 888 Hz; frequency heard by the observer.
20. an observer moves at 24 m/s toward a stationary source emitting 557 Hz; frequency heard by the observer.
21. an observer moves at 57 m/s toward a stationary source emitting 900 Hz; frequency heard by the observer.
22. a stationary source emitting 1657 Hz sends sound to a large flat reflector that moves at 29 m/s toward the source; frequency of the reflected sound received back at the source.
23. a source emitting 1248 Hz moves at 62 m/s toward the observer, and the observer moves at 60 m/s toward the source; frequency heard by the observer.
24. a stationary source emitting 589 Hz sends sound to a large flat reflector that moves at 88 m/s away from the source; frequency of the reflected sound received back at the source.
25. an observer moves at 12 m/s away from a stationary source emitting 1332 Hz; frequency heard by the observer.
26. a stationary source emitting 475 Hz sends sound to a large flat reflector that moves at 30 m/s toward the source; frequency of the reflected sound received back at the source.
27. an observer moves at 19 m/s toward a stationary source emitting 1146 Hz; frequency heard by the observer.
28. a stationary source emitting 1387 Hz sends sound to a large flat reflector that moves at 44 m/s away from the source; frequency of the reflected sound received back at the source.
29. an observer moves at 20 m/s away from a stationary source emitting 1195 Hz; frequency heard by the observer.
30. an observer moves at 19 m/s away from a stationary source emitting 1207 Hz; frequency heard by the observer.
31. a source emitting 958 Hz moves at 17 m/s toward a stationary observer; frequency heard by the observer.
32. a source emitting 801 Hz moves at 56 m/s away from the observer, and the observer moves at 12 m/s toward the source; frequency heard by the observer.
33. an observer moves at 28 m/s toward a stationary source emitting 819 Hz; frequency heard by the observer.
34. an observer moves at 44 m/s away from a stationary source emitting 820 Hz; frequency heard by the observer.
35. an observer moves at 12 m/s away from a stationary source emitting 1902 Hz; frequency heard by the observer.
36. a source emitting 525 Hz moves at 20 m/s toward the observer, and the observer moves at 57 m/s toward the source; frequency heard by the observer.
37. a source emitting 1677 Hz moves at 36 m/s away from a stationary observer; frequency heard by the observer.
38. a source emitting 986 Hz moves at 17 m/s toward a stationary observer; frequency heard by the observer.
39. an observer moves at 51 m/s away from a stationary source emitting 422 Hz; frequency heard by the observer.
40. a source emitting 806 Hz moves at 57 m/s away from the observer, and the observer moves at 55 m/s toward the source; frequency heard by the observer.
41. a source emitting 1192 Hz moves at 8 m/s away from a stationary observer; frequency heard by the observer.
42. a source emitting 544 Hz moves at 32 m/s toward a stationary observer; frequency heard by the observer.
43. a stationary source emitting 1542 Hz sends sound to a large flat reflector that moves at 83 m/s away from the source; frequency of the reflected sound received back at the source.
44. a source emitting 892 Hz moves at 57 m/s away from a stationary observer; frequency heard by the observer.
45. a stationary source emitting 549 Hz sends sound to a large flat reflector that moves at 47 m/s toward the source; frequency of the reflected sound received back at the source.
46. an observer moves at 21 m/s away from a stationary source emitting 1499 Hz; frequency heard by the observer.
47. a source emitting 1731 Hz moves at 19 m/s toward the observer, and the observer moves at 37 m/s toward the source; frequency heard by the observer.

Give the 47 requested frequencies in Hz, each with 4 significant figures, in the order listed, as plain decimal numbers separated by commas.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`723.157,1074.13,346.582,1290.57,641.599,939.167,1138.94,273.662,1984.74,196.938,751.242,1367.8,947.913,1007.92,1833.24,1320.94,2424.11,984.702,686.064,595.974,1049.56,1963.07,1789.84,348.48,1285.4,566.054,1209.48,1071.61,1125.32,1140.14,1007.96,712.669,885.857,714.81,1835.46,650.155,1517.71,1037.42,359.254,801.97,1164.83,599.974,941.127,764.89,723.345,1407.22,2030.19` — проверка: список через запятую (rel_tol 0.002)
