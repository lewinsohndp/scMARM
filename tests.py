import unittest
from ref_query_arm import RefQueryArm
import torch

class TestRefQuery(unittest.TestCase):
    def setUp(self):
        self.ref = torch.tensor([[1,1], [2,2], [10,10],[11,11]])
        self.query = torch.tensor([[1,1],[2,2],[10,10]])
        self.combined = torch.cat([self.ref,self.query], dim=0)

    def test_graph(self):
        # within ref edges (2 per node): [0,0], [1,0], [1,1], [0,1], [2,2], [3,2], [3,3], [2,3]
        # within query edges (2 per node): [4,4], [5,4], [5,5], [4,5], [6,6], [5,6]
        # between ref query edges (1 per ref node): [0,4], [1,5], [2,6], [3, 6]
        
        correct_graph = torch.tensor([[0,1,1,0,2,3,3,2,4,5,5,4,6,5,0,1,2,3], [0,0,1,1,2,2,3,3,4,4,5,5,6,6,4,5,6,6]])
        test_arm = RefQueryArm("configs/semi_basic_linear.txt", 2, 1, ref_batch_size=4)
        test_graph = test_arm.construct_graph(self.combined)

        self.assertTrue(torch.equal(correct_graph, test_graph))

if __name__ == '__main__':
    unittest.main() 