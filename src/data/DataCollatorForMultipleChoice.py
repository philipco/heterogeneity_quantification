from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union

import torch


@dataclass
class DataCollatorForMultipleChoice:
    '''
    This class is designed to handle the formatting and batching the data for mutiple-choice tasks.

    From https://www.kaggle.com/code/lusfernandotorres/kaggle-llm-science-exam-eda-deberta/input.
    '''

    # The tokenizer to be used for tokenizing the data
    tokenizer: PreTrainedTokenizerBase

    # The strategy to be used for padding the data
    padding: Union[bool, str, PaddingStrategy] = True

    # The maximum length for any input sequence
    max_length: Optional[int] = None

    # If provided, pad the sequences to a multiple of this value
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        print("features", features)

        # Finding the correct label key in the features
        label_name = "label" if 'label' in features[0].keys() else 'labels'

        # Extracting the labels and removing them from features
        labels = [feature.pop(label_name) for feature in features]

        # Obtaining batch size
        batch_size = len(features)

        # Obtaining number of choices
        num_choices = len(features[0]['input_ids'])

        # Reestructuring features so each question-choice pair becomes a separate example
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        # Padding all sequences to the same length
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        # Reshaping the batch back into the original format
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

        # Adding the labels back into the batch as a tensor
        batch['labels'] = torch.tensor(labels,
                                       dtype=torch.int64)

        # Returning the batch
        return batch