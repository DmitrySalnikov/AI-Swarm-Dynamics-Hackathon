# Пакет задач TheoremQA: механика

- id: `ladder_rung_2/mechanics`
- источник: TheoremQA, physics, subfields Kinetics, Classic/Celestial/Fluid mechanics; problem ids in task data; лицензия: MIT
- генератор: `r2d_tqa_mechanics`, number of problems in the bundle p = 4, сид 2000
- отбор: медиана 10796 ток., CV 0.07, верно 5/5

## Условие

```text
Below are 4 independent physics problems on mechanics (kinematics, dynamics, gravitation, fluids). Solve each of them.

Problem 1.
Incompressible oil of density 850 kg/m^3 is pumped through a cylindrical pipe at a rate of 9.5 liters per second. The second section of the pipe has a diameter of 4.0 cm. What are the flow speed in that section? (Unit: m/s)
[Answer unit: m/s]

Problem 2.
Passengers on a carnival ride move at constant speed in a horizontal circle of radius 5.0 m, making a complete circle in 4.0 s. What is their acceleration? (Unit: m/s^2))
[Answer unit: m/s^2]

Problem 3.
The mass of one of the small spheres of a Cavendish balance is 0.0100 kg, the mass of the nearest large sphere is 0.500 kg, and the center-to-center distance between them is 0.0500 m. Assuming the gravitational force on each sphere due to the other is $X * 10^{-10}$ N, what is X?
[Answer unit: give the number X]

Problem 4.
The asteroid Pallas has an orbital period of 4.62 years and an orbital eccentricity of 0.233. Find the semi-major axis of its orbit. (Unit: 10^11 m)
[Answer unit: units of 10^11 m]

Give the final answer as a comma-separated list of 4 numbers, one per problem in order (Problem 1 first). Write each number as a plain decimal or in e-notation (e.g. 6.2e-5), without units and without thousands separators.

Think it through, then finish with a final line of the form `ANSWER: <your answer>`.
```

## Эталон

`7.6,12,1.33,4.15` — проверка: список через запятую (rel_tol 0.04)
