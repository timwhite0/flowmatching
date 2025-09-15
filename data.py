import lightning
from sklearn.datasets import make_circles
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split


class CirclesDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        circles_factor: float,
        circles_noise: float,
        num_batches: int,
        batch_size: int,
        train_split: float,
        val_split: float,
    ):
        super().__init__()
        self.circles_factor = circles_factor
        self.circles_noise = circles_noise
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split

    def setup(self, stage: str):
        z, x = make_circles(
            n_samples=self.num_batches * self.batch_size,
            factor=self.circles_factor,
            noise=self.circles_noise,
        )

        dataset = TensorDataset(Tensor(z), Tensor(x).unsqueeze(-1))

        size_total = len(dataset)
        size_train = int(self.train_split * size_total)
        size_val = int(self.val_split * size_total)
        size_test = size_total - size_train - size_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [size_train, size_val, size_test]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
