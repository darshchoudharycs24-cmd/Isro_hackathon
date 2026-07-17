"""
Loads the trained CycleGAN generator (G_A: cloudy -> clean) and runs
reconstruction on a single image. Bridges trained model -> evaluation code.
"""

import sys
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "pytorch-CycleGAN-and-pix2pix"))
from models.networks import define_G


class CloudRemovalModel:
    def __init__(self, checkpoint_path, gpu_id=-1):
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu")

        self.netG = define_G(
            input_nc=3, output_nc=3, ngf=64,
            netG="resnet_9blocks", norm="instance",
            use_dropout=False, init_type="normal", init_gain=0.02,
        )

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.netG.load_state_dict(state_dict)
        self.netG.to(self.device)
        self.netG.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _preprocess(self, cv2_img):
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (256, 256))
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        return tensor

    def _postprocess(self, tensor, original_size):
        img = tensor.squeeze(0).detach().cpu().numpy()
        img = (img * 0.5 + 0.5) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (original_size[1], original_size[0]))
        return img_bgr

    def reconstruct(self, cloudy_cv2_img):
        original_size = cloudy_cv2_img.shape[:2]
        input_tensor = self._preprocess(cloudy_cv2_img)
        with torch.no_grad():
            output_tensor = self.netG(input_tensor)
        return self._postprocess(output_tensor, original_size)


if __name__ == "__main__":
    CHECKPOINT = "pytorch-CycleGAN-and-pix2pix/checkpoints/cloud_cyclegan/latest_net_G_A.pth"
    TEST_IMAGE = "data/sample.jpg"

    if not os.path.exists(CHECKPOINT):
        print(f"[!] Checkpoint not found at {CHECKPOINT}")
        print("    Train the model first (see train_model.sh / README steps)")
    else:
        model = CloudRemovalModel(CHECKPOINT, gpu_id=-1)
        cloudy_img = cv2.imread(TEST_IMAGE)
        result = model.reconstruct(cloudy_img)
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/gan_reconstructed.png", result)
        print("Saved reconstruction to outputs/gan_reconstructed.png")