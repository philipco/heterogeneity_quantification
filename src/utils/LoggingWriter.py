from torch.utils.tensorboard import SummaryWriter
import pickle
import os

class LoggingWriter(SummaryWriter):
    """
    A custom TensorBoard writer that logs scalars and histograms and allows retrieval, saving, and loading.
    """

    def __init__(self, log_dir=None):
        """
        Initialize the LoggingWriter.

        Args:
            log_dir (str, optional): Directory to store logs.
        """
        super().__init__(log_dir)
        self.scalars = {}
        self.histograms = {}
        self.log_dir = log_dir if log_dir else "runs/default"

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        """
        Log a scalar value.

        Args:
            tag (str): The tag associated with the scalar.
            scalar_value (float): The value to log.
            global_step (int, optional): The step at which the value is logged.
        """
        super().add_scalar(tag, scalar_value, global_step, *args, **kwargs)

        if tag not in self.scalars:
            self.scalars[tag] = {}
        self.scalars[tag][global_step] = scalar_value

    def add_histogram(self, tag, values, global_step=None, *args, **kwargs):
        """
        Log a histogram of values.

        Args:
            tag (str): The tag associated with the histogram.
            values (list or numpy array): The values to log.
            global_step (int, optional): The step at which the values are logged.
        """
        super().add_histogram(tag, values, global_step, *args, **kwargs)

        if tag not in self.histograms:
            self.histograms[tag] = {}
        self.histograms[tag][global_step] = values

    def get_scalar(self, tag, global_step):
        """
        Retrieve a logged scalar value.

        Args:
            tag (str): The tag associated with the scalar.
            global_step (int): The step at which the value was logged.

        Returns:
            float or None: The retrieved scalar value or None if not found.
        """
        return self.scalars.get(tag, {}).get(global_step, None)

    def get_histogram(self, tag, global_step):
        """
        Retrieve a logged histogram of values.

        Args:
            tag (str): The tag associated with the histogram.
            global_step (int): The step at which the values were logged.

        Returns:
            list or None: The retrieved histogram values or None if not found.
        """
        return self.histograms.get(tag, {}).get(global_step, None)

    def retrieve_information(self, field):
        """
        Retrieve values and steps for a given field from stored scalars.

        Args:
            field (str): The name of the scalar field to retrieve.

        Returns:
            tuple: A tuple (values, steps) where values are the logged values of the field,
                   and steps are the corresponding steps at which they were recorded.
        """
        if field in self.scalars:
            steps = sorted(self.scalars[field].keys())
            values = [self.scalars[field][step].item() for step in steps]
            return steps, values
        return [], []

    def retrieve_histogram_information(self, field):
        """
        Retrieve values and steps for a given field from stored histograms.

        Args:
            field (str): The name of the histogram field to retrieve.

        Returns:
            tuple: A tuple (values, steps) where values are the logged histogram values of the field,
                   and steps are the corresponding steps at which they were recorded.
        """
        if field in self.histograms:
            steps = sorted(self.histograms[field].keys())
            values = [self.histograms[field][step] for step in steps]
            return steps, values
        return [], []

    def save(self, path, name: str = "logging_writer_central.pkl"):
        """
        Save the writer's state to a file.

        Args:
            path (str): The directory where the state will be saved.
        """
        with open(os.path.join(path, name), "wb") as f:
            pickle.dump({'scalars': self.scalars, 'histograms': self.histograms}, f)

    @staticmethod
    def load(path, name: str = "logging_writer_central.pkl"):
        """
        Load a LoggingWriter from a saved file.

        Args:
            path (str): The directory where the state is stored.

        Returns:
            LoggingWriter: The loaded writer instance.
        """
        writer = LoggingWriter()
        pkl_path = os.path.join(path, name)
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                writer.scalars = data['scalars']
                writer.histograms = data['histograms']
        return writer
