import torch
from models.MLP import MLP
from torchviz import make_dot

model = MLP(input_size=784, hidden_sizes=[512, 256, 128], num_classes=10)

dummy_input = torch.randn(1, 1, 28, 28)

output = model(dummy_input)

dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("model_viz")

print("Model saved.")