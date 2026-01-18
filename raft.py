import torch
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch.nn.functional as nnF
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


class OpticalFlowCalculator:
    def __init__(self, device=None):
        self.device = device
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_large(weights=self.weights, progress=False).to(self.device)
        self.model.eval()
        self.to_tensor = T.ToTensor()

    def preprocess(self, img1, img2, transforms):
        target_height, target_width = 512, 512

        img1_resized = F.resize(img1, size=[target_height, target_width], antialias=False)
        img2_resized = F.resize(img2, size=[target_height, target_width], antialias=False)

        from torchvision.transforms import GaussianBlur


        gaussian_blur = GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))


        img1_blurred = gaussian_blur(img1_resized)
        img2_blurred = gaussian_blur(img2_resized)

        img1_transformed, img2_transformed = transforms(img1_blurred, img2_blurred)

        img1_batch = img1_transformed.unsqueeze(0)
        img2_batch = img2_transformed.unsqueeze(0)

        return img1_batch, img2_batch, (target_height, target_width)

    def resize_flow(self, flow_tensor, target_size, mode='bilinear', align_corners=False):
        n, c, h, w = flow_tensor.shape
        target_h, target_w = target_size

        scale_h = target_h / h
        scale_w = target_w / w

        resized_flow = nnF.interpolate(
            flow_tensor,
            size=target_size,
            mode=mode,
            align_corners=align_corners
        )

        resized_flow[:, 0, :, :] *= scale_w
        resized_flow[:, 1, :, :] *= scale_h

        return resized_flow

    def calculate_optical_flow(self, img1, img2):

        img1_pil = img1
        img2_pil = img2
        orig_height, orig_width = img1_pil.size[1], img1_pil.size[0]

        img1_tensor = self.to_tensor(img1_pil)
        img2_tensor = self.to_tensor(img2_pil)

        img1_batch, img2_batch, _ = self.preprocess(img1_tensor, img2_tensor, self.transforms)

        img1_batch = img1_batch.to(self.device)
        img2_batch = img2_batch.to(self.device)

        with torch.no_grad():
            flow_predictions = self.model(img1_batch, img2_batch)

        flow = flow_predictions[-1]

        flow_resized = self.resize_flow(flow, (orig_height, orig_width))

        return flow_resized

