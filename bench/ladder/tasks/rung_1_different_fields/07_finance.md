# Остаток по кредиту с досрочными платежами

- id: `ladder_rung_1/finance`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r1_fin_loan`, number of months = 84, сид 1000
- отбор: медиана 17996 ток., CV 0.13, верно 5/5

## Условие

```text
A loan has an outstanding balance of $53947.92 at the start of month 1. The nominal annual interest rate is 7.80%, i.e. the monthly interest rate is 0.65%. Every month the following happens, in this order:
  1. interest = (balance at the start of the month) × (monthly rate in effect for that month), rounded to the nearest cent (half a cent rounds up);
  2. the interest is added to the balance and the regular payment of $445.00 is subtracted;
  3. if an extra prepayment is scheduled for that month, it is subtracted as well.
The balance after step 3 is the balance at the start of the next month. The regular payment stays $445.00 for the whole period. Scheduled events:
- At the end of month 3, after the regular payment, an extra prepayment of $1280.00 is made.
- At the end of month 8, after the regular payment, an extra prepayment of $2394.09 is made.
- At the end of month 10, after the regular payment, an extra prepayment of $215.00 is made.
- Starting with month 26, the monthly interest rate becomes 0.55% (annual 6.60%).
- Starting with month 27, the monthly interest rate becomes 0.50% (annual 6.00%).
- Starting with month 32, the monthly interest rate becomes 0.45% (annual 5.40%).
- Starting with month 36, the monthly interest rate becomes 0.40% (annual 4.80%).
- At the end of month 39, after the regular payment, an extra prepayment of $2137.00 is made.
- Starting with month 41, the monthly interest rate becomes 0.50% (annual 6.00%).
- At the end of month 57, after the regular payment, an extra prepayment of $1858.98 is made.
- At the end of month 70, after the regular payment, an extra prepayment of $722.00 is made.
- Starting with month 79, the monthly interest rate becomes 0.60% (annual 7.20%).
- Starting with month 81, the monthly interest rate becomes 0.75% (annual 9.00%).
- Starting with month 82, the monthly interest rate becomes 0.65% (annual 7.80%).

What is the outstanding balance at the end of month 84 (after all steps of month 84)? Give the amount in dollars with exactly two decimals, without the $ sign and without thousands separators (e.g. 12345.67).

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`26960.19` — проверка: число, точное совпадение
