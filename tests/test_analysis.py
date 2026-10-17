import unittest
import numpy as np
from swarm_pipeline.analyze import windows,vector_key,phase_rows,metrics

class AnalysisTests(unittest.TestCase):
    def test_pairwise_cosine_matches_direct_off_diagonal_average(self):
        rng=np.random.default_rng(42)
        for n in [2,4,10]:
            X=rng.normal(size=(n,16));X/=np.linalg.norm(X,axis=1,keepdims=True)
            result=metrics([{'x':0,'phase':'solving','X':X}])[0]
            gram=X@X.T
            expected=gram[~np.eye(n,dtype=bool)].mean()
            self.assertAlmostEqual(result['pair_cosine'],expected,places=7)
            self.assertEqual(result['n_active'],n)

    def test_v2_has_one_commit_point_without_a_fabricated_second_round(self):
        agents=[];vectors={}
        for i in range(3):
            phases={
                'negotiation':{'prefix':0,'offsets':np.array([(k,k+1) for k in range(4)]),'turns':[{'text':f'preference{i}','x':1},{'text':f'claim{i}','x':3}]},
                'solving':{'prefix':4,'text':'x'*20,'offsets':np.array([(k,k+1) for k in range(20)])},
                'exchange':{'prefix':24,'offsets':np.array([(k,k+1) for k in range(4)]),'turns':[{'text':f'commit{i}','x':26}]}}
            agents.append({'protocol':'v2','phases':phases})
            texts=[r['text'] for phase in ['negotiation','exchange'] for r in phases[phase]['turns']]
            texts += [r['text'] for r in windows(phases['solving']['text'],phases['solving']['offsets'],8)]
            for text in texts:vectors[vector_key(text)]=np.array([1.,0.,0.])
        rows,spans=phase_rows(agents,8,vectors)
        self.assertEqual(sum(r['phase']=='negotiation' for r in rows),2)
        self.assertEqual(sum(r['phase']=='solving' for r in rows),5)
        self.assertEqual(sum(r['phase']=='exchange' for r in rows),1)
        self.assertEqual(rows[-1]['X'].shape,(3,3))

if __name__=='__main__':unittest.main()
