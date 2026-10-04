"""Concatenate everything the model generated into long text files, one folder per set.

    python3 embedding_eval/build_concat.py            # writes embedding_eval/concat/<set>/...

concat/<set>/sequential/run_<k>.txt   all 10 solutions of run k, bench order: CoT + answer, CoT + answer, ...
concat/<set>/shuffled/run_<k>.txt     same solutions, question order shuffled per file (seeded, recorded in index)
concat/<set>/group/run_<k>/agent_<i>.txt
                                      agent i of chat run k, in time order: negotiation turns (reasoning + message),
                                      its solve (reasoning + response), merge turns (reasoning + message)
Each .txt holds only generated text, pieces separated by a blank line; <name>.index.json next to it gives the
character span [start, end) and origin of every piece.
"""
import json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOLO = HERE / "runs" / "deepseek-v4-flash-0731"            # run_2 ... run_20
RUN1 = ROOT / "results" / "deepseek-v4-flash-0731"          # run_1
CHAT = HERE / "runs" / "chat"
OUT = HERE / "concat"
SETS, N_RUNS, SEP = ("set_3", "set_4"), 20, "\n\n"


def write(path, pieces):
    """pieces: list of (text, meta). Empty texts are skipped (e.g. a cut-off solve has no response)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out, index, pos = [], [], 0
    for text, meta in pieces:
        if not text:
            continue
        if out:
            out.append(SEP); pos += len(SEP)
        out.append(text)
        index.append({**meta, "start": pos, "end": pos + len(text)})
        pos += len(text)
    path.write_text("".join(out))
    path.with_suffix(".index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    return pos


def solo(bench, k, domain):
    stem = (RUN1 / bench / domain) if k == 1 else (SOLO / f"run_{k}" / bench / domain)
    rec = json.loads(Path(f"{stem}.json").read_text())
    cot = Path(f"{stem}.reasoning.txt").read_text()
    assert cot.strip(), f"empty CoT: {stem}"
    return cot, rec["response"].strip(), rec["finish_reason"]


def solo_pieces(bench, k, domains):
    pieces = []
    for d in domains:
        cot, ans, finish = solo(bench, k, d)
        meta = {"task": f"{bench}/{d}", "run": k, "finish_reason": finish}
        pieces += [(cot, {**meta, "part": "cot"}), (ans, {**meta, "part": "answer"})]
    return pieces


def turns(run_dir, agent):
    path = run_dir / f"agent_{agent}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    for bench in SETS:
        domains = [json.loads(l)["domain"] for l in open(ROOT / "bench" / f"{bench}.jsonl") if l.strip()]
        base = OUT / bench
        # 1, 2: sequential and shuffled solo files
        for k in range(1, N_RUNS + 1):
            write(base / "sequential" / f"run_{k}.txt", solo_pieces(bench, k, domains))
            order = domains[:]
            random.Random(f"{bench}-shuffle-run_{k}").shuffle(order)
            write(base / "shuffled" / f"run_{k}.txt", solo_pieces(bench, k, order))
        # 3: group files, one per agent per chat run
        n_group = 0
        for neg in sorted((CHAT / bench / "negotiation").glob("run_*"), key=lambda p: int(p.name.split("_")[1])):
            if "_invalid_" in neg.name:
                continue
            mrg = CHAT / bench / "merge" / neg.name
            assignment = json.loads((neg / "result.json").read_text())
            tr = json.loads((mrg / "transcript.json").read_text())
            assert tr["agent_tasks"] == {a: t for a, t in assignment.items() if a != "valid"}, f"assignment mismatch {neg}"
            for i in range(1, tr["N"] + 1):
                pieces = []
                for phase, d in (("negotiation", neg), ("merge", mrg)):
                    if phase == "merge":
                        s = json.loads((mrg / "solve" / f"agent_{i}.json").read_text())
                        meta = {"phase": "solve", "task": s["task_id"], "finish_reason": s["finish_reason"]}
                        pieces += [(s["reasoning"].strip(), {**meta, "part": "reasoning"}),
                                   (s["response"].strip(), {**meta, "part": "response"})]
                    for r in turns(d, i):
                        meta = {"phase": phase, "turn": r["turn"], "round": r["round"]}
                        pieces += [(r["reasoning"].strip(), {**meta, "part": "reasoning"}),
                                   (r["message"].strip(), {**meta, "part": "message"})]
                write(base / "group" / neg.name / f"agent_{i}.txt", pieces)
                n_group += 1
        print(f"{bench}: sequential {N_RUNS}, shuffled {N_RUNS}, group {n_group} files -> {base}")


if __name__ == "__main__":
    main()
