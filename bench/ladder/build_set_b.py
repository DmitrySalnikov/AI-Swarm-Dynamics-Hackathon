"""Set B of the set_v2 ladder: ten more tasks per rung for the covert-channel experiment (no model calls).

Every set-B task is a new instance of the generator of the set-A task with the same id suffix: same generator, same
length parameter, a new seed (set-A seed + SEED_SHIFT, the next one if the prompt repeats a set-A or set-B prompt).
So every rung keeps its structure (rung 1: ten knowledge areas, rung 2: ten physics sections, rung 3: ten linear
algebra task types, rung 4: ten matrices of one determinant template). The length calibration holds per generator and
parameter; unlike set A, the single instances were not re-run and selected. Answers are recomputed by each generator's
independent verify() and graded with grade(); a mismatch stops the build.

  uv run --with sympy --with numpy --with scipy --with reasoning-gym==0.1.19 \\
      python bench/ladder/build_set_b.py artifacts/pipeline-ladder-v2/items artifacts/pipeline-ladder-v2/items-b
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'generators'), str(HERE.parents[1]/'embedding_eval')]
from common import build  # noqa: E402
from gen_runs import grade  # noqa: E402
from selftest import all_modules  # noqa: E402

SEED_SHIFT = 10000


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    gens = {g.name: g for m in all_modules() for g in m.GENS}
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob('rung_*.jsonl')):
        rung = path.stem.split('_')[1]
        items = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        seen = {a['prompt'] for a in items}
        set_name = f'ladder_v2b_rung_{rung}'
        rows = []
        for a in items:
            gen, p, seed = gens[a['gen']['name']], a['gen']['value'], a['gen']['seed'] + SEED_SHIFT
            while True:
                b = build(gen, p, seed, set_name, task_id=a['domain'])
                if b['prompt'] not in seen:
                    break
                seed += 1
            if not grade(b, gen.verify(b['data'])):
                raise SystemExit(f"{b['id']}: verify() disagrees with the generated answer")
            seen.add(b['prompt'])
            b['title'] = a['title']
            b['selection'] = {'set_a_task': a['id'], 'rule': 'same generator and length parameter as the set-A task, new seed'}
            del b['data']
            rows.append(b)
        (out/path.name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        print(path.name, [(r['domain'], r['gen']['name'], r['gen']['seed']) for r in rows])


if __name__ == '__main__':
    main()
