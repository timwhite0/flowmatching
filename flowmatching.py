import lightning
import torch
from torch import nn, optim

from nets import VelocityNet, tEncoder, xEncoder


class FlowMatchingModule(lightning.LightningModule):
    def __init__(self, t_embed_dim, z_dim, x_embed_dim, velo_net_channels, lr):
        super().__init__()

        self.lr = lr

        self.t_encoder = tEncoder(t_embed_dim)
        self.x_encoder = xEncoder(x_embed_dim)
        self.velocity_net = VelocityNet(
            z_dim, x_embed_dim, t_embed_dim, velo_net_channels
        )

        self.loss = nn.MSELoss()

    def compute_loss(self, batch, batch_idx, mode):
        # sample z1 and x
        z1, x = batch
        batch_size = z1.shape[0]

        # sample z0
        z0 = torch.randn_like(z1, device=z1.device)

        # encode x
        x_embedding = self.x_encoder(x)

        # encode t
        t = torch.rand([batch_size, 1], device=z1.device)
        t_embedding = self.t_encoder(t)

        # construct zt
        zt = (1 - t) * z0 + t * z1

        # compute true velocity
        u_true = z1 - z0

        # predict velocity
        u_pred = self.velocity_net(zt, x_embedding, t_embedding)

        # compute loss
        loss = self.loss(u_pred, u_true)
        self.log(
            f"{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, mode="test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
