from .classification import ClassificationModel, evaluate
import torch.nn as nn

def get_model_and_loss(task_type, num_classes):
    if task_type == "classification":
        model = ClassificationModel(num_classes)
        loss_fn = nn.CrossEntropyLoss()
        eval_fn = evaluate
    else:
        raise ValueError("Unsupported task type")
    return model, loss_fn, eval_fn