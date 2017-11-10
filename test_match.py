import unittest
import numpy as np
import tracker


class TestMatch(unittest.TestCase):
    def test_oneUnmatched(self):
        distance_matrix = np.array([
            [0.01, 0.2, 14],
            [2, 3, 2],
            [0.1, 0.4, 0.09],
        ])
        matches = tracker.greedy_similarity_match(distance_matrix, similarity_threshold=1)

        self.assertEqual(matches, [(0, 0), (2, 2)])

    def test_containingSuperLargeValue(self):
        # Note: the value cannot actually be Inf, because then greedy_similarity_match gets stuck in an infinite loop
        distance_matrix = np.array([
            [0.01, 0.2, 14],
            [2, 1_000_000_000, 2],
            [0.1, 0.4, 0.09],
        ])
        matches = tracker.greedy_similarity_match(distance_matrix, similarity_threshold=1)

        self.assertEqual(matches, [(0, 0), (2, 2)])


if __name__ == '__main__':
    unittest.main()
