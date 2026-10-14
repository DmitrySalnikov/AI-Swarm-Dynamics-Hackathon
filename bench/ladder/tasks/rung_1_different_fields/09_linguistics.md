# Сортировка слов по выдуманному алфавиту

- id: `ladder_rung_1/linguistics`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r1_ling_alphabet`, number of words = 79, сид 1000
- отбор: медиана 19821 ток., CV 0.16, верно 5/5

## Условие

```text
An invented language uses only the following 12 letters, and its alphabetical order is (from first to last):

  b < n < z < o < i < p < t < u < e < r < k < d

Words are compared letter by letter from the left: at the first position where they differ, the word whose letter comes earlier in this alphabet comes first. (No word in the list is a beginning of another.)

Sort the following 79 words in increasing alphabetical order of this language:

kbozpuz, bozore, kndub, doneti, tzodu, kbi, tiu, kboez, kir, kndp, bubob, ziper, potib, ponib, kbozour, koi, boriru, zitin, tzopk, kboepzrr, nuritzi, ponetd, zizzui, kurii, kize, kbkor, rzrn, ponkbk, nuribeb, zipd, kboeood, porik, tzke, rubobuk, tedoi, potinp, poneb, dozu, kndudu, kbok, ribuz, teno, konikb, zitio, zin, rido, bunou, ponorbn, buzizuo, ponozi, kiden, kut, poneu, koe, zipnn, kotri, tio, kiori, rinbt, kod, kuniti, ponuzz, tekzud, bir, kokn, kuuzk, kop, tzkn, zipipt, rit, kbrer, zir, tenpdpe, kond, bopo, ribbbei, nee, potiz, bipi

Give the sorted words separated by commas, first word first.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`bozore,bopo,boriru,bipi,bir,bubob,bunou,buzizuo,nuribeb,nuritzi,nee,zin,zizzui,zipnn,zipipt,ziper,zipd,zitin,zitio,zir,ponozi,ponorbn,ponib,ponuzz,poneb,ponetd,poneu,ponkbk,potib,potinp,potiz,porik,tzopk,tzodu,tzkn,tzke,tio,tiu,teno,tenpdpe,tekzud,tedoi,rzrn,ribbbei,ribuz,rinbt,rit,rido,rubobuk,kbozour,kbozpuz,kboez,kboeood,kboepzrr,kbok,kbi,kbrer,kbkor,kndp,kndub,kndudu,konikb,kond,koi,kop,kotri,koe,kokn,kod,kize,kiori,kir,kiden,kuniti,kut,kuuzk,kurii,doneti,dozu` — проверка: список через запятую
