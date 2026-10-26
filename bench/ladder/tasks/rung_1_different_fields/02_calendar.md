# День недели для григорианских дат

- id: `ladder_rung_1/calendar`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r1_cal_weekday`, number of dates = 64, сид 1000
- отбор: медиана 18244 ток., CV 0.17, верно 5/5

## Условие

```text
For each of the following dates in the Gregorian calendar (extended to all years shown; leap years are years divisible by 4, except years divisible by 100 that are not divisible by 400), find the day of the week.

1. 16 September 2400
2. 30 September 2400
3. 1 July 1954
4. 16 September 2253
5. 13 October 2246
6. 22 January 2098
7. 18 July 2187
8. 7 March 1882
9. 2 June 1812
10. 9 July 1812
11. 26 February 1798
12. 20 October 2193
13. 20 March 1961
14. 13 June 2240
15. 5 April 2206
16. 27 February 1913
17. 10 February 2243
18. 14 December 1676
19. 18 July 2095
20. 12 August 2231
21. 12 May 1811
22. 27 July 2193
23. 13 June 1948
24. 7 November 2051
25. 8 November 1928
26. 17 December 2089
27. 5 December 2280
28. 19 October 2264
29. 14 September 1820
30. 22 November 2210
31. 2 August 2006
32. 25 November 2125
33. 15 January 1920
34. 13 June 1671
35. 15 December 2084
36. 22 October 2225
37. 16 November 2049
38. 31 January 1954
39. 28 August 1859
40. 16 March 2249
41. 8 August 2071
42. 2 January 2057
43. 29 September 1616
44. 2 January 2312
45. 23 November 2069
46. 12 July 2106
47. 8 July 2138
48. 10 February 1886
49. 7 July 1908
50. 26 April 2265
51. 26 January 1634
52. 17 April 1973
53. 1 August 2393
54. 30 March 1732
55. 10 April 1828
56. 8 April 2379
57. 17 May 1865
58. 9 January 2160
59. 4 January 2225
60. 3 August 2296
61. 23 December 2054
62. 29 March 1726
63. 12 April 2262
64. 27 October 1616

Give the 64 weekday names in English (Monday, Tuesday, …), separated by commas, in the order of the dates.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`Saturday,Saturday,Thursday,Friday,Tuesday,Wednesday,Wednesday,Tuesday,Tuesday,Thursday,Monday,Sunday,Monday,Saturday,Saturday,Thursday,Friday,Monday,Monday,Friday,Sunday,Saturday,Sunday,Tuesday,Thursday,Saturday,Sunday,Wednesday,Thursday,Thursday,Wednesday,Sunday,Thursday,Saturday,Friday,Saturday,Tuesday,Sunday,Sunday,Friday,Saturday,Tuesday,Thursday,Tuesday,Saturday,Monday,Tuesday,Wednesday,Tuesday,Wednesday,Thursday,Tuesday,Sunday,Sunday,Thursday,Sunday,Wednesday,Wednesday,Tuesday,Monday,Wednesday,Friday,Saturday,Thursday` — проверка: список через запятую
