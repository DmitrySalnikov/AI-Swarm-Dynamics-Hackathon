# Расшифровка шифра Виженера по ключу

- id: `ladder_rung_1/cryptography`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r1_crypto_vigenere`, ciphertext length, letters = 80, сид 1000
- отбор: медиана 16004 ток., CV 0.18, верно 5/5

## Условие

```text
A message (English words written together, letters only, no spaces) was encrypted with the Vigenère cipher. Letters are numbered A = 0, B = 1, …, Z = 25; the key is repeated over the whole message, and the i-th ciphertext letter is C_i = (P_i + K_i) mod 26, where K_i is the key letter at position i (so decryption is P_i = (C_i − K_i) mod 26).

Key: CAOOPU

Ciphertext (80 letters, written in groups of five only for readability; the key continues across group boundaries):
YHOHG IQFYB DQVHO HBYGT WBIIY HOHGY UTKSH NHOZZ DQDRW BVPQI QSUUN LQFDM UOTHT HULSS EYPTS FLITK

Decrypt the message. Give the plaintext as one string of uppercase letters without spaces.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`WHATROOFKNOWTHATMEETINTOWHATRESTWESTFOLLOWBRINGVOICEFALLCROSSOFTENSLEEPENTERWORK` — проверка: строка совпадает
