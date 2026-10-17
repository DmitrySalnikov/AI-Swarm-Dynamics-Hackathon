import unittest
from swarm_pipeline.answer_format import canonical_answer

class AnswerFormatTests(unittest.TestCase):
    def test_v2_numeric_submit_completes_without_format_retry(self):
        import tempfile,json
        from pathlib import Path
        from swarm_pipeline.runner import Config
        from swarm_pipeline.protocol_v2 import V2Runner,V2OfflineClient
        from swarm_pipeline.audit import audit
        class NumericClient(V2OfflineClient):
            def complete(self,q):
                r=super().complete(q)
                if q['context']['phase']=='solving':
                    a=json.loads(r['content']);a['answer']=-970;r['content']=json.dumps(a)
                return r
        tasks=[{'id':f't{i}','prompt':'Give an integer.','answer':'-970','check':{'type':'exact'}} for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            r=V2Runner(tasks,Config(condition='cooperative',agents=2,offline=True),Path(tmp),NumericClient(),workers=2)
            r.run();report=audit(r.directory)
            self.assertEqual(report['action_errors'],[])
            self.assertTrue(report['single_successful_submit_and_commit_per_agent'])
            self.assertTrue(all(list(a['answers'].values())==['-970'] for a in r.state['agents'].values()))

    def test_numbers_preserve_answers_in_canonical_text(self):
        self.assertEqual(canonical_answer(-970),'-970')
        self.assertEqual(canonical_answer(2.2361),'2.2361')
        self.assertEqual(canonical_answer('16037/123'),'16037/123')

    def test_invalid_types_do_not_become_answers(self):
        for value in [True,None,[],{},float('nan'),float('inf')]:
            with self.assertRaises(ValueError):canonical_answer(value)

if __name__=='__main__':unittest.main()
