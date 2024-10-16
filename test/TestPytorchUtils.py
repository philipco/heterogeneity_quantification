import pytest
import torch
import torch.nn as nn
from typing import List

from src.optim.PytorchUtilities import aggregate_models


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
