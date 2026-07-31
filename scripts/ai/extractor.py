import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine as cosine_distance
from torchvision import transforms
import openvino as ov

from scripts.config import PATH_TO_OSNET_MODEL, PATH_TO_OSNET_OPENVINO_XML_MODEL
from scripts.ai.osnet import osnet_x0_25, osnet_x1_0


class ReIDExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        #self.model = osnet_x0_25(num_classes=1041, pretrained=False)
        self.model = osnet_x1_0(num_classes=1041, pretrained=False)
        checkpoint = torch.load(PATH_TO_OSNET_MODEL, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract(self, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.squeeze().cpu().numpy()

    @staticmethod
    def similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        return 1.0 - float(cosine_distance(feat_a, feat_b))



class ReIDExtractorOpenVINO:
    def __init__(self):
        self.core = ov.Core()

        self.compiled_model = self.core.compile_model(
            PATH_TO_OSNET_OPENVINO_XML_MODEL,
            "CPU",
        )

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def extract(self, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb)
        # OpenVINO espera numpy
        input_tensor = tensor.unsqueeze(0).numpy()
        result = self.compiled_model(
            {self.input_layer: input_tensor}
        )

        feat = result[self.output_layer].squeeze()

        # L2 normalization (igual ao PyTorch)
        feat /= np.linalg.norm(feat)

        return feat.astype(np.float32)

    @staticmethod
    def similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        return 1.0 - float(cosine_distance(feat_a, feat_b))