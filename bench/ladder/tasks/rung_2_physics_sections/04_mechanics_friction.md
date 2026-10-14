# Брусок на трассе с трением

- id: `ladder_rung_2/mechanics_friction`
- источник: own generator (answer computed by code); лицензия: CC BY-NC-SA 4.0 (this repository)
- генератор: `r2_friction_track`, number of segments N = 19, сид 2000
- отбор: медиана 18272 ток., CV 0.17, верно 5/5

## Условие

```text
A small block starts at the beginning of segment 1 with speed 12 m/s and slides along a track made of 19 consecutive straight segments (in the direction of motion):
1. downhill incline at 14° to the horizontal, length 5.8 m (measured along the surface), kinetic friction coefficient μ = 0.19
2. downhill incline at 7° to the horizontal, length 8.6 m (measured along the surface), kinetic friction coefficient μ = 0.10
3. horizontal, length 7.4 m (measured along the surface), kinetic friction coefficient μ = 0.15
4. downhill incline at 30° to the horizontal, length 3.1 m (measured along the surface), kinetic friction coefficient μ = 0.24
5. downhill incline at 29° to the horizontal, length 5.0 m (measured along the surface), kinetic friction coefficient μ = 0.11
6. downhill incline at 27° to the horizontal, length 6.6 m (measured along the surface), kinetic friction coefficient μ = 0.38
7. downhill incline at 28° to the horizontal, length 3.2 m (measured along the surface), kinetic friction coefficient μ = 0.36
8. horizontal, length 4.4 m (measured along the surface), kinetic friction coefficient μ = 0.22
9. downhill incline at 32° to the horizontal, length 6.6 m (measured along the surface), kinetic friction coefficient μ = 0.34
10. downhill incline at 20° to the horizontal, length 11.5 m (measured along the surface), kinetic friction coefficient μ = 0.26
11. horizontal, length 9.4 m (measured along the surface), kinetic friction coefficient μ = 0.15
12. downhill incline at 30° to the horizontal, length 3.9 m (measured along the surface), kinetic friction coefficient μ = 0.13
13. downhill incline at 11° to the horizontal, length 10.9 m (measured along the surface), kinetic friction coefficient μ = 0.38
14. horizontal, length 6.6 m (measured along the surface), kinetic friction coefficient μ = 0.23
15. downhill incline at 33° to the horizontal, length 3.6 m (measured along the surface), kinetic friction coefficient μ = 0.14
16. uphill incline at 23° to the horizontal, length 11.0 m (measured along the surface), kinetic friction coefficient μ = 0.14
17. downhill incline at 20° to the horizontal, length 10.9 m (measured along the surface), kinetic friction coefficient μ = 0.16
18. downhill incline at 25° to the horizontal, length 5.4 m (measured along the surface), kinetic friction coefficient μ = 0.35
19. uphill incline at 25° to the horizontal, length 3.6 m (measured along the surface), kinetic friction coefficient μ = 0.23

"Uphill" means the block moves up the incline, "downhill" means it moves down. The block stays in contact with the surface, passes every junction without any loss of speed, and there is no air resistance. Use g = 9.8 m/s². It is guaranteed that the block never stops on the track.

Find the speed of the block at the end of the last segment, in m/s, with 4 significant figures.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`10.3037` — проверка: число (rel_tol 0.002)
