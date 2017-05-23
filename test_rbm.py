import unittest
from rbm import RBM
import numpy as np
import itertools


class TestRBM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_run(self):
        r = RBM(num_visible=6, num_hidden=2)
        training_data = np.array(
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0]])
        r.fit(training_data, max_epochs=1000)
        user = np.array([[0, 0, 0, 1, 1, 0]])
        r.run_visible(user)

    def test_predict(self):
        rbm = RBM(num_visible=3, num_hidden=1)
        # xor
        train_data = np.asarray([[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
        rbm.fit(train_data, max_epoch=5000)
        print(rbm.run_visible([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]))
        print(rbm.predict([[0, 0, 0], [0, 0, 1]]))
        print(rbm.predict([[0, 1, 0], [0, 0, 1]]))
        print(rbm.predict([[1, 0, 0], [0, 0, 1]]))
        print(rbm.predict([[1, 1, 0], [0, 0, 1]]))


if __name__ == '__main__':
    unittest.main()
