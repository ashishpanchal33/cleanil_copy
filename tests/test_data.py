import unittest
import torch
from cleanil.data import load_d4rl_replay_buffer

class TestData(unittest.TestCase):
    def test_load_d4rl_replay_buffer(self):
        """Test d4rl non-expert trajectory masks have been removed so that data has no all zeros"""
        buffer = load_d4rl_replay_buffer("hopper-medium-expert-v2", batch_size=256)
        data = buffer._storage

        num_zeros = torch.all(data["observation"] == 0, dim=-1).sum()
        self.assertTrue(num_zeros == 0)


if __name__ == "__main__":
    unittest.main()