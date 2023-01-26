import pytorch_lightning as pl
import rul_datasets


def checkpoint(approach, ckpt_path, max_rul=50):
    dm = rul_datasets.RulDataModule(
        rul_datasets.reader.DummyReader(1, max_rul=max_rul), 32
    )
    trainer = pl.Trainer(max_epochs=0, num_sanity_val_steps=0)
    trainer.fit(approach, dm)
    trainer.save_checkpoint(ckpt_path)
