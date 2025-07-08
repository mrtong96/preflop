import unittest

import numpy as np

from src.post_flop_solver.post_flop_utils import get_vector_index, VECTOR_SIZE, get_cards_from_vector_index


class TestPostflopUtils(unittest.TestCase):
    def test_get_vector_index(self):
        # I care less that the numbers are right and more that it functions correctly
        vector_values = set()
        for i in range(52):
            for j in range(i + 1, 52):
                i, j = np.int8(i), np.int8(j)
                # symmetry
                self.assertEqual(get_vector_index(i, j), get_vector_index(j, i))
                result = get_vector_index(i, j)
                # result in expected range
                self.assertTrue(0 <= result < VECTOR_SIZE)
                # result isn't the same as another i/j pair
                self.assertTrue(result not in vector_values)
                vector_values.add(result)

                # test that inverse works too
                inverse_result = get_cards_from_vector_index(result)
                self.assertEqual(inverse_result[0], i)
                self.assertEqual(inverse_result[1], j)
