from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only


class MLFlowModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                run_id = logger.run_id
                # print(f"{run_id = }")
                # print(f"{self.best_model_path = }")
                if self.best_model_path:
                    logger.experiment.log_artifact(run_id, self.best_model_path)
