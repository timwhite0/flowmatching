import hydra
import lightning
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    lightning.seed_everything(cfg.seed)

    datamodule = instantiate(cfg.datamodule)

    flowmodule = instantiate(cfg.flowmodule)

    trainer = instantiate(cfg.trainer)

    trainer.fit(model=flowmodule, datamodule=datamodule)
    trainer.test(model=flowmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
