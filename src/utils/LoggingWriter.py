from torch.utils.tensorboard import SummaryWriter


class LoggingWriter(SummaryWriter):
    def __init__(self, log_dir=None):
        super().__init__(log_dir)
        self.scalars = {}

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        # Log the scalar as usual
        super().add_scalar(tag, scalar_value, global_step, *args, **kwargs)

        # Store the value for later retrieval
        if tag not in self.scalars:
            self.scalars[tag] = {}
        self.scalars[tag][global_step] = scalar_value

    def get_scalar(self, tag, global_step):
        # Retrieve the scalar value
        return self.scalars.get(tag, {}).get(global_step, None)
