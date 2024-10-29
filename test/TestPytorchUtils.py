import pytest
import torch
import torch.nn as nn
from typing import List

from src.optim.PytorchUtilities import aggregate_models, load_new_model


# Dummy model class for testing
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# Test 1: Test that aggregation works correctly with equal weights
def test_aggregation_equal_weights():
    device = 'cpu'
    main_model_idx = 0
    model1 = DummyModel().to(device)
    model2 = DummyModel().to(device)
    model3 = DummyModel().to(device)

    models = [model1, model2, model3]
    weights = [1, 1, 1]

    # Set identical initial parameters for all models for simplicity
    for model in models:
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

    aggregate_models(main_model_idx, models, weights, device)

    for param in models[main_model_idx].parameters():
        assert torch.all(param == 3.0), "Parameters should be sum of the 3 models with equal weights"


# Test 2: Test that aggregation works with different weights
def test_aggregation_different_weights():
    device = 'cpu'
    main_model_idx = 0
    model1 = DummyModel().to(device)
    model2 = DummyModel().to(device)
    model3 = DummyModel().to(device)

    models = [model1, model2, model3]
    weights = [2, 3, 1]

    # Set identical initial parameters for all models
    for model in models:
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

    aggregate_models(main_model_idx, models, weights, device)

    # With weights 2, 3, and 1, the total sum should be 6
    for param in models[main_model_idx].parameters():
        assert torch.all(param == 6.0), "Parameters should be weighted sum (2+3+1) of models"


# Test 3: Test that aggregation does not affect models other than idx_main_model
def test_no_change_to_other_models():
    device = 'cpu'
    main_model_idx = 0
    model1 = DummyModel().to(device)
    model2 = DummyModel().to(device)
    model3 = DummyModel().to(device)

    models = [model1, model2, model3]
    weights = [1, 1, 1]

    # Save original parameters for non-main models
    model2_params_before = [param.clone() for param in model2.parameters()]
    model3_params_before = [param.clone() for param in model3.parameters()]

    aggregate_models(main_model_idx, models, weights, device)

    # Ensure model2 and model3 parameters are unchanged
    for param_before, param_after in zip(model2_params_before, model2.parameters()):
        assert torch.equal(param_before, param_after), "Parameters of model2 should remain unchanged"
    for param_before, param_after in zip(model3_params_before, model3.parameters()):
        assert torch.equal(param_before, param_after), "Parameters of model3 should remain unchanged"


# Test 4: Test aggregation on CUDA (if available)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_aggregation_on_cuda():
    device = 'cuda'
    main_model_idx = 0
    model1 = DummyModel().to(device)
    model2 = DummyModel().to(device)
    model3 = DummyModel().to(device)

    models = [model1, model2, model3]
    weights = [1, 1, 1]

    # Set identical initial parameters for all models
    for model in models:
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

    aggregate_models(main_model_idx, models, weights, device)

    for param in models[main_model_idx].parameters():
        assert torch.all(param == 3.0), "Parameters should be aggregated correctly on CUDA"


# Test 1: Test if parameters are correctly copied
def test_parameters_copied_correctly():
    model_to_update = DummyModel()
    new_model = DummyModel()

    # Set specific parameters in the new_model
    with torch.no_grad():
        for param in new_model.parameters():
            param.fill_(2.0)

    load_new_model(model_to_update, new_model)

    # Verify all parameters in model_to_update match those in new_model
    for param_to_update, param_new in zip(model_to_update.parameters(), new_model.parameters()):
        assert torch.all(param_to_update == 2.0), "Parameters should match those of new_model"


# Test 2: Test if model_to_update parameters change while new_model stays the same being modified.
def test_new_model_unchanged():
    model_to_update = DummyModel()
    new_model = DummyModel()

    # Set specific parameters in both models
    with torch.no_grad():
        for param in model_to_update.parameters():
            param.fill_(1.0)
        for param in new_model.parameters():
            param.fill_(2.0)


    load_new_model(model_to_update, new_model)

    # Set specific parameters in both models
    with torch.no_grad():
        for param in new_model.parameters():
            param.fill_(3.0)


    # Verify parameters of new_model remain unchanged
    for param in new_model.parameters():
        assert torch.all(param == 3.0), "Parameters of new_model should be equal to 3"

    # Verify parameters of new_model remain unchanged
    for param in model_to_update.parameters():
        assert torch.all(param == 2.0), "Parameters of new_model should remain unchanged"


# Test 3: Test error when models have different architectures
def test_error_different_architectures():
    model_to_update = DummyModel()
    different_model = nn.Linear(5, 2)  # Different architecture

    with pytest.raises(KeyError):
        load_new_model(model_to_update, different_model)


import pytest
import torch
from src.optim.PytorchUtilities import aggregate_gradients


# Test 1: Aggregation with equal weights
def test_aggregation_equal_weights():
    gradients_list = [
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    ]
    weights = [1, 1, 1]

    result = aggregate_gradients(gradients_list, weights)

    # Expected result: sum of each gradient multiplied by its weight
    expected = [torch.tensor([3.0, 6.0]), torch.tensor([9.0, 12.0])]
    for res, exp in zip(result, expected):
        assert torch.all(res == exp), f"Expected {exp}, but got {res}"


# Test 2: Aggregation with different weights
def test_aggregation_different_weights():
    gradients_list = [
        [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0, 2.0])],
        [torch.tensor([2.0, 2.0]), torch.tensor([2.0, 2.0, 3.0])],
        [torch.tensor([3.0, 3.0]), torch.tensor([3.0, 3.0, 4.0])]
    ]
    weights = [0.1, 0.2, 0.7]

    result = aggregate_gradients(gradients_list, weights)

    # Expected result: weighted sum of gradients
    expected = [torch.tensor([2.6, 2.6]), torch.tensor([2.6, 2.6, 3.6])]
    for res, exp in zip(result, expected):
        assert torch.all(res == exp), f"Expected {exp}, but got {res}"


# Test 3: Check if aggregation does not modify input gradients
def test_no_modification_of_input_gradients():
    gradients_list = [
        [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])],
        [torch.tensor([2.0, 2.0]), torch.tensor([2.0, 2.0])],
        [torch.tensor([3.0, 3.0]), torch.tensor([3.0, 3.0])]
    ]
    weights = [1, 1, 1]

    # Make deep copies of the input gradients
    original_gradients = [[grad.clone() for grad in grads] for grads in gradients_list]

    # Run the aggregation
    _ = aggregate_gradients(gradients_list, weights)

    # Verify each original gradient is unchanged
    for original, current in zip(original_gradients, gradients_list):
        for orig_grad, curr_grad in zip(original, current):
            assert torch.equal(orig_grad, curr_grad), "Input gradients should remain unchanged"
