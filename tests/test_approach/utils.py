import pytorch_lightning as pl
import rul_datasets


def checkpoint(approach, ckpt_path):
    dm = rul_datasets.RulDataModule(rul_datasets.reader.DummyReader(1), 32)
    trainer = pl.Trainer(max_epochs=0, num_sanity_val_steps=0)
    trainer.fit(approach, dm)
    trainer.save_checkpoint(ckpt_path)
