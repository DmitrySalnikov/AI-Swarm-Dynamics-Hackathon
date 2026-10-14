# Изображение в системе тонких линз

- id: `ladder_rung_2/geometric_optics`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r2_thin_lenses`, number of lenses N = 100, сид 2001
- отбор: медиана 13721 ток., CV 0.23, верно 5/5

## Условие

```text
A coaxial optical system consists of 100 thin lenses in air, numbered 1 to 100 in the direction of light propagation (left to right). A point object is on the axis 21 cm to the left of lens 1.

lens 1: f = 28 cm; distance from lens 1 to lens 2: 14 cm
lens 2: f = 49 cm; distance from lens 2 to lens 3: 38 cm
lens 3: f = 20 cm; distance from lens 3 to lens 4: 25 cm
lens 4: f = 6 cm; distance from lens 4 to lens 5: 50 cm
lens 5: f = -14 cm; distance from lens 5 to lens 6: 44 cm
lens 6: f = -27 cm; distance from lens 6 to lens 7: 39 cm
lens 7: f = 38 cm; distance from lens 7 to lens 8: 10 cm
lens 8: f = -40 cm; distance from lens 8 to lens 9: 9 cm
lens 9: f = 37 cm; distance from lens 9 to lens 10: 30 cm
lens 10: f = -22 cm; distance from lens 10 to lens 11: 48 cm
lens 11: f = 46 cm; distance from lens 11 to lens 12: 14 cm
lens 12: f = -39 cm; distance from lens 12 to lens 13: 30 cm
lens 13: f = 27 cm; distance from lens 13 to lens 14: 6 cm
lens 14: f = 45 cm; distance from lens 14 to lens 15: 28 cm
lens 15: f = 9 cm; distance from lens 15 to lens 16: 44 cm
lens 16: f = 46 cm; distance from lens 16 to lens 17: 33 cm
lens 17: f = -40 cm; distance from lens 17 to lens 18: 8 cm
lens 18: f = 32 cm; distance from lens 18 to lens 19: 44 cm
lens 19: f = 38 cm; distance from lens 19 to lens 20: 43 cm
lens 20: f = -22 cm; distance from lens 20 to lens 21: 26 cm
lens 21: f = 46 cm; distance from lens 21 to lens 22: 29 cm
lens 22: f = -36 cm; distance from lens 22 to lens 23: 6 cm
lens 23: f = -60 cm; distance from lens 23 to lens 24: 41 cm
lens 24: f = -44 cm; distance from lens 24 to lens 25: 26 cm
lens 25: f = 18 cm; distance from lens 25 to lens 26: 49 cm
lens 26: f = 18 cm; distance from lens 26 to lens 27: 31 cm
lens 27: f = -34 cm; distance from lens 27 to lens 28: 50 cm
lens 28: f = 59 cm; distance from lens 28 to lens 29: 24 cm
lens 29: f = -47 cm; distance from lens 29 to lens 30: 46 cm
lens 30: f = 40 cm; distance from lens 30 to lens 31: 34 cm
lens 31: f = 22 cm; distance from lens 31 to lens 32: 6 cm
lens 32: f = -6 cm; distance from lens 32 to lens 33: 22 cm
lens 33: f = -40 cm; distance from lens 33 to lens 34: 30 cm
lens 34: f = 60 cm; distance from lens 34 to lens 35: 46 cm
lens 35: f = 38 cm; distance from lens 35 to lens 36: 20 cm
lens 36: f = -24 cm; distance from lens 36 to lens 37: 23 cm
lens 37: f = 40 cm; distance from lens 37 to lens 38: 30 cm
lens 38: f = -37 cm; distance from lens 38 to lens 39: 32 cm
lens 39: f = 53 cm; distance from lens 39 to lens 40: 49 cm
lens 40: f = -38 cm; distance from lens 40 to lens 41: 18 cm
lens 41: f = 11 cm; distance from lens 41 to lens 42: 16 cm
lens 42: f = 6 cm; distance from lens 42 to lens 43: 8 cm
lens 43: f = 36 cm; distance from lens 43 to lens 44: 13 cm
lens 44: f = -58 cm; distance from lens 44 to lens 45: 16 cm
lens 45: f = 30 cm; distance from lens 45 to lens 46: 20 cm
lens 46: f = -21 cm; distance from lens 46 to lens 47: 14 cm
lens 47: f = -44 cm; distance from lens 47 to lens 48: 29 cm
lens 48: f = 34 cm; distance from lens 48 to lens 49: 50 cm
lens 49: f = 52 cm; distance from lens 49 to lens 50: 12 cm
lens 50: f = 35 cm; distance from lens 50 to lens 51: 5 cm
lens 51: f = -6 cm; distance from lens 51 to lens 52: 8 cm
lens 52: f = -11 cm; distance from lens 52 to lens 53: 34 cm
lens 53: f = 7 cm; distance from lens 53 to lens 54: 42 cm
lens 54: f = 17 cm; distance from lens 54 to lens 55: 24 cm
lens 55: f = -20 cm; distance from lens 55 to lens 56: 34 cm
lens 56: f = 7 cm; distance from lens 56 to lens 57: 37 cm
lens 57: f = 46 cm; distance from lens 57 to lens 58: 5 cm
lens 58: f = 34 cm; distance from lens 58 to lens 59: 37 cm
lens 59: f = -40 cm; distance from lens 59 to lens 60: 46 cm
lens 60: f = 50 cm; distance from lens 60 to lens 61: 20 cm
lens 61: f = -30 cm; distance from lens 61 to lens 62: 9 cm
lens 62: f = 30 cm; distance from lens 62 to lens 63: 25 cm
lens 63: f = -20 cm; distance from lens 63 to lens 64: 12 cm
lens 64: f = 56 cm; distance from lens 64 to lens 65: 28 cm
lens 65: f = -21 cm; distance from lens 65 to lens 66: 46 cm
lens 66: f = 37 cm; distance from lens 66 to lens 67: 11 cm
lens 67: f = -36 cm; distance from lens 67 to lens 68: 33 cm
lens 68: f = 36 cm; distance from lens 68 to lens 69: 37 cm
lens 69: f = -18 cm; distance from lens 69 to lens 70: 14 cm
lens 70: f = -38 cm; distance from lens 70 to lens 71: 36 cm
lens 71: f = 56 cm; distance from lens 71 to lens 72: 47 cm
lens 72: f = -40 cm; distance from lens 72 to lens 73: 39 cm
lens 73: f = -26 cm; distance from lens 73 to lens 74: 7 cm
lens 74: f = 58 cm; distance from lens 74 to lens 75: 6 cm
lens 75: f = 32 cm; distance from lens 75 to lens 76: 40 cm
lens 76: f = -56 cm; distance from lens 76 to lens 77: 6 cm
lens 77: f = -18 cm; distance from lens 77 to lens 78: 46 cm
lens 78: f = 41 cm; distance from lens 78 to lens 79: 26 cm
lens 79: f = -28 cm; distance from lens 79 to lens 80: 24 cm
lens 80: f = 16 cm; distance from lens 80 to lens 81: 16 cm
lens 81: f = -8 cm; distance from lens 81 to lens 82: 26 cm
lens 82: f = 6 cm; distance from lens 82 to lens 83: 32 cm
lens 83: f = 46 cm; distance from lens 83 to lens 84: 18 cm
lens 84: f = 32 cm; distance from lens 84 to lens 85: 34 cm
lens 85: f = 6 cm; distance from lens 85 to lens 86: 18 cm
lens 86: f = 26 cm; distance from lens 86 to lens 87: 12 cm
lens 87: f = -38 cm; distance from lens 87 to lens 88: 41 cm
lens 88: f = 36 cm; distance from lens 88 to lens 89: 24 cm
lens 89: f = -22 cm; distance from lens 89 to lens 90: 15 cm
lens 90: f = 16 cm; distance from lens 90 to lens 91: 33 cm
lens 91: f = 6 cm; distance from lens 91 to lens 92: 23 cm
lens 92: f = -20 cm; distance from lens 92 to lens 93: 8 cm
lens 93: f = -6 cm; distance from lens 93 to lens 94: 42 cm
lens 94: f = 23 cm; distance from lens 94 to lens 95: 26 cm
lens 95: f = 20 cm; distance from lens 95 to lens 96: 7 cm
lens 96: f = -6 cm; distance from lens 96 to lens 97: 10 cm
lens 97: f = 5 cm; distance from lens 97 to lens 98: 46 cm
lens 98: f = -33 cm; distance from lens 98 to lens 99: 28 cm
lens 99: f = 30 cm; distance from lens 99 to lens 100: 11 cm
lens 100: f = -32 cm

Sign convention: for every lens use 1/s + 1/s' = 1/f, with f > 0 for a converging and f < 0 for a diverging lens; s is the object distance, positive when the object is to the left of the lens (real object) and negative when it is to the right of the lens (virtual object, i.e. the light arrives converging toward a point behind the lens); s' is the image distance, positive when the image is to the right of the lens (real) and negative when it is to the left (virtual). The image formed by each lens is the object for the next lens (if that image lies beyond the next lens, it is a virtual object for it); ignore apertures and aberrations.

Find the position of the final image formed by lens 100: its signed distance from lens 100 in cm (positive if it is to the right of lens 100, negative if to the left). Give the exact value as an integer or a reduced fraction.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`-64` — проверка: число, точное совпадение
