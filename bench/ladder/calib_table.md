| Генератор | Ступени | Параметр | Точки: значение → длина | Модель | p* | Выбор | Замечание |
|---|---|---|---|---|---|---|---|
| la_det | 1,3,4 | matrix size n | 5 → 11792✓; 5 → 12794✓; 6 → 32275✓; 8 → 72551✂ | pow k=3.81 | 5.59 | 6 |  |
| la_det_e4 | 1,3,4 | matrix size n | 6 → 17683✓; 6 → 15583✓ | prop | 6.79 | 7 |  |
| la_det_e20 | 1,3,4 | matrix size n | 5 → 32378✓; 5 → 26473✓ | prop | 3.78 | 4 |  |
| bio_gc | 1,4 | sequence length L, nt | 200 → 5582✓; 300 → 5148✗; 300 → 5544✓; 400 → 43042✓ | pow k=2.65 | 392 | 400 |  |
| r1d_sara_tax | 1 | number of tax cases p | 1 → 2644✓; 3 → 4888✓°; 16 → 16507✓°; 16 → 13910✓° | lin a=2104 b=820 | 21.8 | 16 | above grid |
| r1d_finmath | 1 | number of problems in the bundle p | 4 → 12032✓; 8 → 15554✗; 12 → 22362✗; 12 → 10138✓° | lin a=10565 b=495 | 19.1 | 12 | above grid |
| r1d_zebra | 1 | number of houses = number of characteristics p | 5 → 6362✓; 6 → 18663✓; 6 → 22702✓; 6 → 17423✓ | pow k=6.14 | 6.03 | 6 |  |
| r1d_calendar | 1 | number of questions K | 10 → 7567✓°; 24 → 10689✓°; 66 → 25377✓; 66 → 25156✓° | lin a=3702 b=325 | 50.1 | 50 |  |
| r1d_shortest_paths | 1 | number of vertices N | 20 → 7274✓; 38 → 15519✓; 38 → 13238✓; 40 → 21548✓ | pow k=1.27 | 44.9 | 44 |  |
| r1_crypto_dh | 1 | prime modulus p | 401 → 7161✓; 701 → 14978✓; 701 → 15746✓; 1009 → 39424✓ | pow k=1.79 | 755 | 809 |  |
| r1_crypto_vigenere | 1 | ciphertext length, letters | 80 → 18343✓; 80 → 19842✓; 100 → 24637✓; 260 → 53486✗ | lin a=4532 b=189 | 81.9 | 80 |  |
| r1_crypto_rsa | 1 | number of ciphertexts | 3 → 5037✓; 8 → 8260✓; 26 → 18284✓°; 26 → 14326✓ | lin a=3979 b=476 | 33.7 | 30 |  |
| r1_cal_weekday | 1 | number of dates | 5 → 3103✓; 15 → 5676✓; 71 → 16006✓°; 71 → 28119✓° | lin a=1510 b=289 | 63.9 | 64 |  |
| r1_cal_roman | 1 | number of dates | 6 → 1268✓°; 16 → 3095✓; 109 → 8363✗; 109 → 9563✓ | lin a=1385 b=70 | 267 | 160 | above grid |
| r1_music_just | 1 | number of notes | 15 → 5930✓; 40 → 16328✓; 49 → 18516✓; 49 → 12454✓ | lin a=2266 b=289 | 61.4 | 61 |  |
| r1_nt_order | 1 | number of (a, m) pairs | 3 → 1851✓; 8 → 7137✓; 17 → 15739✓; 17 → 22389✓ | pow k=1.33 | 17.7 | 18 |  |
| r1_cs_stack | 1 | number of instructions | 50 → 4305✓; 130 → 5965✓; 810 → 40897✓; 810 → 29163✓ | lin a=1455 b=41 | 448 | 450 |  |
| r1_geom_polygon | 1 | number of vertices | 24 → 3352✓; 60 → 6453✓; 200 → 20036✓; 200 → 23798✓ | lin a=446 b=107 | 183 | 183 |  |
| r1_bio_translation | 1 | number of codons | 80 → 1891✓; 200 → 3980✓; 1120 → 39138✓; 1120 → 23478✗ | pow k=1.08 | 780 | 780 |  |
| r1_gen_selection | 1 | number of generations | 5 → 14290✓; 11 → 15844✓; 11 → 29735✓; 14 → 22721✓ | lin a=10010 b=1038 | 9.63 | 10 |  |
| r1_chem_mixture | 1 | number of samples | 3 → 12972✓; 5 → 17138✓; 5 → 19452✓; 7 → 28025✓ | lin a=580 b=3763 | 5.16 | 5 |  |
| r1_chem_chain | 1 | number of steps | 6 → 25453✓; 14 → 20463✓; 14 → 31996✓; 16 → 23570✓ | prop | 13.7 | 14 |  |
| r1_chem_buffer | 1 | number of buffer solutions | 4 → 2237✓; 10 → 3486✓; 89 → 20423✓; 89 → 18594✓ | lin a=1440 b=203 | 91.4 | 91 |  |
| r1_fin_loan | 1 | number of months | 12 → 4187✓; 30 → 7167✓; 108 → 22470✓; 108 → 28667✓ | lin a=961 b=227 | 83.8 | 84 |  |
| r1_fin_irr | 1 | number of cash flows after t=0 | 4 → 20109✓; 4 → 17627✓; 5 → 22419✓; 12 → 45439✓ | lin a=5680 b=3315 | 4.32 | 4 |  |
| r1_econ_cournot | 1 | number of firms | 40 → 6928✓; 120 → 13812✓; 192 → 24484✓; 192 → 18898✓ | lin a=2699 b=98 | 176 | 176 |  |
| r1_econ_dwl | 1 | number of linear segments | 15 → 10117✓; 45 → 12740✓; 120 → 17048✓; 120 → 16292✓ | lin a=9566 b=60 | 175 | 120 | above grid |
| r1_ling_numerals | 1 | number of words to decode | 6 → 5768✓; 21 → 3659✓; 21 → 4299✓; 25 → 4987✓ | prop | 20.8 | 21 |  |
| r1_ling_alphabet | 1 | number of words | 12 → 4337✓; 30 → 8006✓; 89 → 22431✓; 89 → 22144✓ | lin a=1256 b=236 | 79.5 | 79 |  |
| r1_stat_ols | 1 | number of data points | 15 → 7480✓; 40 → 9738✓; 150 → 16126✓; 150 → 22063✓ | lin a=6247 b=86 | 160 | 150 |  |
| r2d_tqa_mechanics | 2 | number of problems in the bundle p | 2 → 3463✓; 4 → 14561✓; 5 → 24560✗°; 5 → 60119✂° | pow k=2.58 | 4.05 | 4 |  |
| r2d_tqa_modern | 2 | number of problems in the bundle p | 2 → 6824✓; 4 → 15621✗°; 5 → 17323✗; 5 → 18496✗ | pow k=1.06 | 5.41 | 5 |  |
| r2d_tqa_thermal | 2 | number of problems in the bundle p | 2 → 18318✓; 2 → 9807✓; 2 → 8991✓; 4 → 10819✓ | prop | 2.18 | 2 |  |
| r2d_ugp_thermodynamics | 2 | number of problems in the bundle p | 2 → 43781✓; 3 → 16998✗; 3 → 19188✗; 4 → 25363✗ | prop | 3.13 | 3 |  |
| r2d_ugp_quantum | 2 | number of problems in the bundle p | 2 → 3367✓; 4 → 14330✗; 5 → 36333✗; 5 → 37690✗° | pow k=2.58 | 4.1 | 4 |  |
| r2d_ugp_optics | 2 | number of problems in the bundle p | 2 → 22001✗; 2 → 16229✗; 2 → 6595✗; 4 → 37019✗ | pow k=1.48 | 2.64 | 3 |  |
| r2d_ugp_electromagnetism | 2 | number of problems in the bundle p | 2 → 26020✗; 2 → 2564✗; 2 → 10370✗; 4 → 34259✗ | pow k=1.95 | 3.04 | 3 |  |
| r2d_ugp_atomic | 2 | number of problems in the bundle p | 2 → 11830✗; 3 → 11144✗; 3 → 14640✗; 4 → 30815✗ | pow k=1.26 | 3.54 | 4 |  |
| r2_resistor_sp | 1,2 | number of resistors N | 14 → 2450✓; 32 → 2963✓; 400 → 103143✂; 400 → 75488✂° | pow k=1.15 | 115 | 116 |  |
| r2_ladder_mesh | 2 | number of loops N | 6 → 17858✓; 6 → 32284✓; 6 → 33075✓; 12 → 49846✓ | lin a=5632 b=3684 | 3.9 | 4 |  |
| r2_rc_switching | 2 | number of switching intervals N | 3 → 44544✓; 3 → 23779✓; 10 → 48505✓; 24 → 52815✓ | lin a=33528 b=888 | -15.2 | 3 | below grid |
| r2_thin_lenses | 1,2 | number of lenses N | 10 → 2807✓; 24 → 7265✓; 60 → 13616✓; 60 → 11734✓ | lin a=1743 b=185 | 98.8 | 60 | above grid |
| r2_layer_refraction | 2 | number of layers N | 4 → 24403✓; 4 → 31275✓; 12 → 36341✓; 30 → 63935✂ | lin a=21599 b=1391 | -1.15 | 4 | below grid |
| r2_gas_cycle | 2 | number of processes N | 8 → 33409✓; 11 → 29043✓; 11 → 27186✓; 16 → 29570✓ | prop | 8.09 | 8 |  |
| r2_calorimetry | 2 | number of bodies N | 10 → 12022✓; 22 → 21689✓; 22 → 23528✓; 24 → 21356✓ | lin a=4598 b=772 | 20 | 20 |  |
| r2_level_mean_energy | 2 | number of levels N | 14 → 34806✓; 34 → 20061✓; 34 → 24569✓; 34 → 25676✓ | prop | 33.9 | 34 |  |
| r2_hydrogenlike_lines | 2 | number of transitions K | 12 → 13239✓°; 28 → 14997✓; 28 → 17098✓; 32 → 21736✓° | lin a=8682 b=323 | 35 | 35 |  |
| r2_box_states | 2 | energy bound E_max in units of E1 (actual value p minus 0..9) | 80 → 5248✓; 200 → 15401✓; 250 → 12067✓; 250 → 11164✓ | lin a=3380 b=39 | 427 | 430 |  |
| r2_cart_collisions | 2 | number of collisions K | 8 → 7220✓; 18 → 8612✓; 100 → 46058✓; 100 → 34286✓ | lin a=3182 b=369 | 45.5 | 46 |  |
| r2_friction_track | 2 | number of segments N | 10 → 14999✓; 26 → 18668✗; 32 → 30813✓; 32 → 31336✓ | lin a=6305 b=706 | 19.4 | 19 |  |
| r2_composite_inertia | 2 | number of parts N | 14 → 6516✓; 36 → 9105✓; 120 → 21523✓; 120 → 28644✗ | lin a=3386 b=180 | 92.2 | 92 |  |
| r2_point_charges | 2 | number of charges N | 2 → 32484✓; 2 → 40748✓; 8 → 50346✓°; 20 → 52509✓ | lin a=36958 b=883 | -19.2 | 2 | below grid |
| r2_parallel_wires | 2 | number of wires N | 5 → 13603✓; 5 → 18779✓; 8 → 24127✓; 20 → 42870✓ | lin a=8190 b=1753 | 6.74 | 7 |  |
| r2_decay_chain | 2 | chain length N | 2 → 32187✓; 2 → 32998✓; 4 → 65329✂; 7 → 71678✂ | lin a=20277 b=8072 | -0.0343 | 2 | below grid |
| r2_velocity_addition | 2 | number of frames N | 10 → 2973✓; 28 → 8764✓; 61 → 11420✓; 61 → 23605✓ | lin a=451 b=281 | 69.6 | 70 |  |
| r2_doppler_cases | 2 | number of cases K | 12 → 11524✓; 32 → 18287✓°; 37 → 18031✓°; 37 → 16052✓° | lin a=9055 b=235 | 46.7 | 47 |  |
| r2_pipeline_bernoulli | 2 | number of sections N | 3 → 20774✓; 3 → 7824✓; 12 → 48533✓; 30 → 61919✂ | lin a=13598 b=1764 | 3.63 | 4 |  |
| r2_composite_wall | 2 | number of layers N | 2 → 4214✓; 2 → 7529✓; 15 → 25556✓; 40 → 26899✓ | lin a=7840 b=557 | 21.8 | 22 |  |
| r3_inverse | 3 | matrix size n | 4 → 4970✓; 5 → 5194✓; 9 → 26654✓; 9 → 59370✗ | pow k=2.81 | 7.16 | 7 |  |
| r3_rank | 3 | matrix size n | 6 → 4857✓; 8 → 10226✓; 10 → 18168✓; 10 → 15889✓ | pow k=2.44 | 10.7 | 11 |  |
| r3_eigen | 3 | matrix size n | 3 → 724✓; 3 → 900✓; 4 → 29057✓; 5 → 31795✓ | pow k=7.91 | 4.37 | 4 |  |
| r3_charpoly | 3 | matrix size n | 4 → 4463✓; 5 → 5899✓; 8 → 70157✂; 8 → 78754✂ | pow k=4.37 | 6.01 | 6 |  |
| r3_lu | 3 | matrix size n | 5 → 4443✓; 7 → 7789✓; 10 → 22949✓°; 10 → 18995✓ | pow k=2.29 | 9.96 | 10 |  |
| r3_plu | 3 | matrix size n | 5 → 5696✓; 7 → 11635✓; 9 → 16411✓; 9 → 13634✓ | pow k=1.61 | 10.5 | 9 | above grid |
| r3_solve | 3 | matrix size n | 5 → 2704✓; 7 → 6313✓; 10 → 17522✓; 10 → 24492✓ | pow k=2.98 | 9.96 | 10 |  |
| r3_nullspace | 3 | number of columns n | 6 → 4616✓; 8 → 12137✓; 9 → 15884✓; 9 → 11683✓ | pow k=2.69 | 10.2 | 10 |  |
| r3_rref | 3 | number of rows m (columns m+2) | 4 → 3027✓; 6 → 9214✓; 8 → 26029✗; 8 → 16098✗ | pow k=2.76 | 7.94 | 8 |  |
| r3_power | 3 | matrix size n (power 4) | 3 → 2397✓; 5 → 8360✓; 7 → 25703✗; 7 → 14650✓ | pow k=2.47 | 7.09 | 7 |  |
| r3_gram_schmidt | 3 | number of vectors k (in R^(k+1)) | 3 → 1661✓; 4 → 6904✓; 5 → 9689✓; 5 → 11481✓ | pow k=3.53 | 5.86 | 6 |  |
| r3_projection | 3 | number of spanning vectors k (in R^(k+2)) | 3 → 2832✓; 4 → 4637✓; 7 → 27674✓; 7 → 27576✓ | pow k=2.80 | 6.31 | 6 |  |
| r3_least_squares | 3 | number of unknowns n (n+2 equations) | 3 → 1917✓; 4 → 4968✓; 6 → 13416✓; 6 → 14965✓ | pow k=2.84 | 6.73 | 7 |  |
| r3_adjugate | 3 | matrix size n | 4 → 10280✓; 5 → 14408✓; 6 → 14720✓; 6 → 13799✓ | lin a=3876 b=1795 | 8.98 | 8 |  |
| r3_jordan | 3 | matrix size n | 3 → 1935✓; 5 → 18377✓; 5 → 6362✓; 5 → 5716✓ | pow k=2.95 | 6.62 | 7 |  |
| r3_smith | 3 | matrix size n | 4 → 5423✓; 6 → 21870✓; 6 → 14731✓; 6 → 27783✓ | pow k=3.31 | 5.93 | 6 |  |
| r3_minpoly | 3 | matrix size n | 3 → 3114✓; 5 → 5639✓; 8 → 18285✗; 8 → 10554✓ | pow k=1.55 | 10.3 | 8 | above grid |
| r3_signature | 3 | number of variables n | 5 → 3589✓; 7 → 21582✓; 7 → 24466✓; 7 → 22865✓ | pow k=5.51 | 6.83 | 7 |  |
| r3_cholesky | 3 | matrix size n | 5 → 3379✓; 7 → 4924✓; 10 → 9116✓; 10 → 7229✓ | pow k=1.28 | 20.4 | 10 | above grid |
| r3_coords | 3 | dimension n | 5 → 2760✓; 7 → 9584✓; 9 → 13656✓; 9 → 18012✓ | pow k=2.91 | 9.61 | 10 |  |
| r3_traces | 3 | matrix size n (powers 1..4) | 4 → 5460✓; 5 → 13123✓; 6 → 12557✓; 6 → 9977✓ | pow k=1.62 | 8.09 | 8 |  |
| r3_subspaces | 3 | dimension n | 4 → 7524✓; 6 → 13065✓; 8 → 19049✓; 8 → 25721✓ | pow k=1.57 | 7.58 | 8 |  |
| r3_change_basis | 3 | matrix size n | 3 → 3476✓; 4 → 9007✓; 5 → 12874✓; 5 → 10743✓ | pow k=2.33 | 6.13 | 6 |  |
| r3_qr | 3 | matrix size n | 3 → 3241✓; 4 → 4318✓; 7 → 19319✓; 7 → 23738✓ | pow k=2.37 | 6.92 | 7 |  |
| r3_chain_trace | 3 | number of 3×3 matrices K | 4 → 3471✓; 6 → 4045✓; 62 → 29970✓; 62 → 49501✗ | lin a=609 b=631 | 30.7 | 31 |  |
| r3_gram_det | 3 | number of vectors k (in R^(k+1)) | 4 → 4191✓; 5 → 12779✓; 5 → 9206✓; 5 → 9792✓ | pow k=4.11 | 5.85 | 6 |  |
| r3_svd | 3 | matrix size n | 3 → 1578✓; 4 → 7060✓; 5 → 10229✓; 5 → 9982✓ | pow k=3.53 | 5.91 | 6 |  |
