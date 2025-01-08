import os
import unittest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import KITTIDataset

class TestDataset(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super(TestDataset, self).__init__(methodName)

    def setUp(self):
        self.data_dir = os.path.join("data", "raw")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError

        self.batch_size = 8
        self.augmentation = {
            "random_h_flip": True,
            "color_jitter": True
        }

    def test_dataset_length(self):
        train_set = KITTIDataset(data_dir=self.data_dir, train=True, transforms=None)
        test_set = KITTIDataset(data_dir=self.data_dir, train=False, transforms=None)

        self.assertEqual(int(len(train_set)/len(test_set)), 4, msg="Data split train/test should be 80/20!")
        self.assertGreater(len(train_set), 0, msg="Training dataset length should be greater than 0!")
        self.assertGreater(len(test_set), 0, msg="Validation dataset length should be greater than 0!")

if __name__ == "__main__":
    unittest.main()