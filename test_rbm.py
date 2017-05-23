import unittest
from rbm import RBM
import numpy as np


class TestRBM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_run(self):
        r = RBM(num_input=6, num_hidden=2)
        training_data = np.array(
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0]])
        r.fit(training_data, max_epochs=1000)
        user = np.array([[0, 0, 0, 1, 1, 0]])
        r.run_visible(user)

    def test_predict(self):
        rbm = RBM(num_input=2, num_hidden=1, num_output=2)

        train_data = np.asarray([[0, 0], [1, 1], [0, 1], [1, 0]])
        train_label = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]])

        rbm.fit(train_data, train_label, max_epoch=1000)
        # print(rbm.run_visible([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]))
        print(rbm.free_energy(np.array([[1, 0, 1, 0], [1, 0, 0, 1]])))
        print(rbm.predict(train_data))

    def test_free_energy(self):
        r = RBM(num_input=6, num_hidden=2)
        training_data = np.array(
            [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
             [0, 0, 1, 1, 1, 0]])
        r.fit(training_data, max_epochs=1000)
        f = r.free_energy(np.array([[1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0]]))
        self.assertLess(f[1], f[0])
        f = r.free_energy(np.array([[1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 0]]))
        self.assertLess(f[1], f[0])
        f = r.free_energy(np.array([[1, 0, 1, 0, 1, 1], [0, 0, 1, 1, 1, 0]]))
        self.assertLess(f[1], f[0])
        f = r.free_energy(np.array([[0, 1, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0]]))
        self.assertLess(f[1], f[0])
        f = r.free_energy(np.array([[0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0]]))
        self.assertLess(f[1], f[0])


if __name__ == '__main__':
    unittest.main()
