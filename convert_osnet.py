import torch
import torch.nn as nn
import openvino as ov

from scripts.ai.osnet import osnet_x1_0, osnet_x0_25

# Cria o modelo
model = osnet_x1_0(
    num_classes=1041,
    pretrained=False
)

# Carrega os pesos
state = torch.load(
    "osnet_x1_0_msmt17.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state)

# Remove o classificador
model.classifier = nn.Identity()
model.eval()

# Entrada exemplo
dummy = torch.randn(1, 3, 256, 128)

# Converte
ov_model = ov.convert_model(
    model,
    example_input=dummy
)

# Salva
ov.save_model(ov_model, "osnet_x1_0_openvino.xml")