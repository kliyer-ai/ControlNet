import pytorch_lightning as pl

# from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
# from custom_dataset_cross import MyDataModule
from torch.utils.data import DataLoader

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from kinetics import Kinetics700InterpolateBase

# from pytorch_lightning import seed_everything


# Configs
# resume_path = './models/control-base.ckpt'
resume_path = "./models/control-concat.ckpt"
experiment_name = "kin_hed_concat6"
config_path = "./models/cldm_v15_concat.yaml"

batch_size = 8
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


def main():
    # seed_everything(42)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    seq_time = 0.5
    seq_length = None  # 15

    dataset = Kinetics700InterpolateBase(
        sequence_time=seq_time,
        sequence_length=seq_length,
        size=512,
        resize_size=None,
        random_crop=None,
        pixel_range=2,
        interpolation="bicubic",
        mode="train",
        data_path="/export/compvis-nfs/group/datasets/kinetics-dataset/k700-2020",
        dataset_size=1.0,
        filter_file="/export/home/koktay/flow_diffusion/scripts/timestamps_training.json",
        flow_only=False,
        include_hed=True,
    )
    validation_set = Kinetics700InterpolateBase(
        sequence_time=seq_time,
        sequence_length=seq_length,
        size=512,
        resize_size=None,
        random_crop=None,
        pixel_range=2,
        interpolation="bicubic",
        mode="val",
        data_path="/export/compvis-nfs/group/datasets/kinetics-dataset/k700-2020",
        dataset_size=1.0,
        filter_file="/export/home/koktay/flow_diffusion/scripts/timestamps_validation.json",
        flow_only=False,
        include_hed=True,
    )

    logger = ImageLogger(batch_frequency=logger_freq, name=experiment_name)
    # wandb_logger = WandbLogger(name='kin_hed_cross_2', project="ControlNet")
    # tbl = TensorBoardLogger(save_dir='ControlNet', name='kin_hed_cross_2')

    train_loader = DataLoader(
        dataset, num_workers=4, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        validation_set, num_workers=4, batch_size=batch_size, shuffle=True
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        callbacks=[logger],
        limit_val_batches=1,
        val_check_interval=logger_freq,
        num_sanity_val_steps=2,
        default_root_dir="train_log/" + experiment_name,
    )  # , logger=[wandb_logger, tbl])

    # Train!
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
