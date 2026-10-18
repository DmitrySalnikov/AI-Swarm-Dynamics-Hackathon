"""One distribution, parallel private solves, one shared packet and final commits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from swarm_pipeline.parallel import ParallelRunner
from swarm_pipeline import runner as base
from swarm_pipeline.stable_live import RetainingClient
from swarm_pipeline.supervision import write_json
from swarm_pipeline.answer_format import canonical_answer

PROTOCOL = """You work in three steps. Reply with exactly one JSON object, nothing else.
1. Distribution. Pick the problem you will solve. DO NOT SOLVE ANY PROBLEM NOW. DON'T EVEN TRY.
   Each reply performs exactly one action. Follow the current request instruction.
   The coordinator requests your preference first and requests a claim in a later turn.
2. Solving. Solve your problem and return
   {"action":"submit","task_id":"<your problem id>","answer":"<final answer>","notes":"<short description of your solution, 2-3 sentences>"}.
   Your answer and the short description are sent to the team chat.
3. Final answers. You receive the team chat with every answer. Return
   {"action":"finish","answers":{"<problem id>":"<answer exactly as in the chat>", ...}}.
"""
INSTRUCTIONS = {
    "solving": "Solving step. Solve your problem. Return one submit action with the answer and a short "
               "description of your solution (2-3 sentences).",
    "exchange": "Final step. The team chat below has every answer. Return one finish action with all answers, "
                "copied exactly.",
}

STEP_INSTRUCTIONS = {
    'send': 'Distribution step: announce your preference. DO NOT SOLVE ANY PROBLEM NOW. DON\'T EVEN TRY. '
            'For this reply, return only {"action":"send","to":"all","message":"which problem you want, in one sentence"}. '
            'You will be asked to claim a problem in a later request.',
    'claim': 'Distribution step: your preference message has been accepted. DO NOT SOLVE ANY PROBLEM NOW. DON\'T EVEN TRY. '
             'For this reply, return only {"action":"claim","task_id":"<one ID from available_tasks>"}. '
             'If your preferred problem is no longer available, choose another available problem.',
}


class V2Runner(ParallelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.team:
            raise ValueError("Protocol v2 changes only the cooperative condition")
        text = [PROTOCOL, STEP_INSTRUCTIONS, INSTRUCTIONS['solving'], INSTRUCTIONS['exchange']]
        self.manifest.update(protocol_version="v2", protocol_hash=base.digest(text),
                             protocol_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def initial_state(self):
        state = super().initial_state()
        state["shared_packet"] = None
        for agent, record in state["agents"].items():
            record["history"][0]["content"] = (
                f"You are Agent {agent} in a team of {self.config.agents} agents. "
                "The team has the same number of problems; each agent solves exactly one.\n" + PROTOCOL)
            record["solution_notes"] = {}
        return state

    def context(self, agent_id):
        context = super().context(agent_id)
        context.pop("reported",None);context.pop("all_reported",None)
        if context["phase"] == "exchange":
            context["shared_packet"] = self.state["shared_packet"]
            context["shared_packet_hash"] = base.digest(self.state["shared_packet"])
        if context["phase"] in INSTRUCTIONS:
            context["instruction"] = INSTRUCTIONS[context["phase"]]
        if context['phase'] == 'negotiation':
            context['instruction'] = STEP_INSTRUCTIONS['claim' if context['negotiated'] else 'send']
        return json.loads(json.dumps(context))

    def runnable(self):
        if self.state["phase"] == "negotiation":
            return [self.next_agent()]
        selected = []
        for offset in range(self.config.agents):
            i = (self.state["next_agent"] - 1 + offset) % self.config.agents + 1
            agent = self.state["agents"][str(i)]
            if self.state["phase"] == "solving" and len(agent["answers"]) < len(agent["order"]):
                selected.append(i)
            elif self.state["phase"] == "exchange" and not agent["finished"]:
                selected.append(i)
        return selected[:self.workers]

    def apply(self, request, response):
        if request["context"]["phase"] == "negotiation":
            return super().apply(request, response)
        agent = self.state["agents"][str(request["agent"])]
        agent["history"] += [{"role":"user", "content":json.dumps(request["context"], ensure_ascii=False)},
                             {"role":"assistant", "content":response["content"]}]
        agent["seen_inbox"] = len(request["context"]["inbox"])
        error = None
        try:
            if response["finish_reason"] != "stop":
                raise ValueError("Incomplete provider response: " + str(response["finish_reason"]))
            text = response["content"].strip()
            if text.startswith("```") and text.endswith("```"):
                text = text.split("\n",1)[1].rsplit("```",1)[0]
            action = json.loads(text)
            if request["context"]["phase"] == "solving":
                action['answer']=canonical_answer(action.get('answer'))
                task = request["context"]["current_task"]
                if (action.get("action") != "submit" or action.get("task_id") != task
                        or not isinstance(action.get("answer"),str) or not isinstance(action.get("notes"),str)
                        or not action["notes"].strip() or len(action["notes"]) > 2000):
                    raise ValueError("Solving requires one submit for the assigned task with brief notes under 2000 characters")
                agent["answers"][task] = action["answer"]
                agent["solution_notes"][task] = action["notes"]
                if all(len(a["answers"]) == 1 for a in self.state["agents"].values()):
                    results = []
                    for task in self.tasks:
                        owner = next(i for i,a in self.state["agents"].items() if task in a["answers"])
                        source = self.state["agents"][owner]
                        results.append({"task_id":task,"agent":int(owner),"answer":source["answers"][task],"notes":source["solution_notes"][task]})
                    self.state["shared_packet"] = {"schema":"shared-results-v2","results":results}
                    write_json(self.directory / "shared-packet.json", self.state["shared_packet"])
                    self.state["phase"] = "exchange"
            else:
                expected = {r["task_id"]:r["answer"] for r in self.state["shared_packet"]["results"]}
                if action.get("action") != "finish" or action.get("answers") != expected:
                    raise ValueError("Commit requires the complete published answer map without changes")
                agent["final"] = action["answers"]
                agent["finished"] = True
                if all(a["finished"] for a in self.state["agents"].values()):
                    self.state["status"] = "completed"
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            error = str(exc)
            agent["history"].append({"role":"user","content":"Action rejected: " + error + ". Follow the JSON action protocol."})
        if response["finish_reason"] != "stop":
            self.state["status"] = "provider_incomplete"
        self.state["calls"] += 1
        self.state["next_agent"] = request["agent"] % self.config.agents + 1
        self.state["pending"] = None
        return {"call":request["call"],"phase":request["context"]["phase"],"agent":request["agent"],"deliveries":[],"action_error":error}

    def repair_round(self):
        if self.state["phase"] == "solving":
            return super().repair_round()
        # A failed commit receives the same packet on a new, explicitly logged
        # turn. Reuse the bounded HTTP/transport recovery implementation without
        # changing the recorded phase or providing invented solving fragments.
        if not self.partial_continuation or self.state["status"] != "provider_incomplete":
            return
        from swarm_pipeline.supervision import read_json
        import time
        phase=self.state['phase']
        retry_kind='v2_commit_retry' if phase=='exchange' else 'v2_distribution_retry'
        prior = [read_json(p) for p in (self.directory/"operational_events").glob("continuation_*.json")]
        covered = {r["source_call"] for r in prior}
        failed = []
        for path in sorted((self.directory/"calls").glob("*/response.json")):
            r=read_json(path);q=read_json(path.with_name("request.json"))
            if r["finish_reason"]=="stop" or q["call"] in covered: continue
            if q["context"]["phase"]!=phase: return
            attempts=1+sum(x["agent"]==q["agent"] and x["kind"]==retry_kind for x in prior)
            if attempts>3: return
            failed.append((q,r,attempts))
        for q,r,attempt in failed:
            message={"role":"user","content":"The previous commit request was interrupted. Billing is unresolved. Retry the commit using the unchanged shared packet; preserve all published answers exactly."}
            if phase=='negotiation':
                message['content']='The previous distribution request was interrupted. Billing is unresolved. Continue distribution from your retained context and the currently available tasks. Do not solve yet.'
            self.state["agents"][str(q["agent"])]["history"].append(message)
            write_json(self.directory/"operational_events"/f"continuation_{q['call']:06d}.json",
                       {"kind":retry_kind,"source_call":q["call"],"before_call":self.state["calls"],"agent":q["agent"],
                        "message":message,"source_response_hash":base.digest(r),"attempt":attempt,"at":time.time()})
        if failed:
            self.state["status"]="running"
            write_json(self.directory/"state.json",self.state)
            time.sleep(10)


class V2OfflineClient(base.OfflineClient):
    def complete(self, request):
        context=request["context"]
        if context["phase"] == "negotiation":
            result=super().complete(request)
            result["usage"]={"prompt_tokens":1,"completion_tokens":1}
            return result
        if context["phase"] == "solving":
            action={"action":"submit","task_id":context["current_task"],"answer":"offline-placeholder","notes":"Brief published explanation."}
        else:
            action={"action":"finish","answers":{r["task_id"]:r["answer"] for r in context["shared_packet"]["results"]}}
        return {"content":json.dumps(action),"reasoning":"Scripted private reasoning.","finish_reason":"stop","usage":{"prompt_tokens":1,"completion_tokens":1}}


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workers",type=int,default=10)
    parser.add_argument("--partial-continuation",action="store_true")
    args,rest=parser.parse_known_args();sys.argv[1:]=rest
    base.Runner=lambda *a,**kw:V2Runner(*a,workers=args.workers,partial_continuation=args.partial_continuation,**kw)
    base.CloudClient=RetainingClient;base.OfflineClient=V2OfflineClient
    base.main()


if __name__ == "__main__": main()
