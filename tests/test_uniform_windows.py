import unittest
from types import SimpleNamespace
import numpy as np
from swarm_pipeline.uniform_windows import extract, layouts


def part(text, start):
    return {'text': text, 'start': start, 'end': start + len(text),
            'offsets': np.array([[i, i+1] for i in range(len(text))])}


class UniformWindowTests(unittest.TestCase):
    def test_chat_padding_is_explicit_and_donors_are_not_repeated(self):
        class CharacterTokenizer:
            def encode(self,text,**kwargs):
                return SimpleNamespace(offsets=[(i,i+1) for i in range(len(text))])
        streams=[{'agent':1,'task':'a','phases':{'negotiation':'n','solving':'ss','exchange':'e'}},
                 {'agent':2,'task':'b','phases':{'negotiation':'nnn','solving':'ss','exchange':'eee'}}]
        donors={'a':[{'text':'ABC','sha256':'first'},{'text':'XYZ','sha256':'second'}]}
        result,spans,provenance=layouts(streams,CharacterTokenizer(),'chat_supplement',donors)
        self.assertEqual(extract(result[1],0,3),'n\n\nAB')
        self.assertEqual(extract(result[1],5,3),'e\n\nXY')
        self.assertEqual(extract(result[1],3,2),'ss')
        self.assertEqual([p['phase'] for p in provenance],['negotiation','exchange'])
        self.assertEqual(len({p['donor']['sha256'] for p in provenance}),2)

    def test_adjacent_real_segments_can_cross_boundary(self):
        self.assertEqual(extract([part('abcd', 0), part('efgh', 4)], 2, 4), 'cd\n\nef')

    def test_no_crossing_gap_or_repeating_tail(self):
        parts = [part('abcd', 0), part('efgh', 8)]
        self.assertIsNone(extract(parts, 2, 4))
        self.assertIsNone(extract(parts, 5, 4))
        self.assertIsNone(extract(parts, 10, 4))
        self.assertEqual(extract(parts, 8, 4), 'efgh')


if __name__ == '__main__':
    unittest.main()
