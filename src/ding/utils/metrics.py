"""
Evaluation metrics for image/video inpainting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import linalg

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
import clip


###########################
# Helpers
###########################

class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img
            
def to_tensors():
    return transforms.Compose([Stack(), ToTorchFormatTensor()])


###########################
# I3D models
###########################


def init_i3d_model(i3d_model_path):
    print(f"[Loading I3D model from {i3d_model_path} for FID score ..]")
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    i3d_model.load_state_dict(torch.load(i3d_model_path, weights_only=False))
    return i3d_model


def get_i3d_activations(batched_video,
                        i3d_model,
                        target_endpoint='Logits',
                        flatten=True,
                        grad_enabled=False):
    """
    Get features from i3d model and flatten them to 1d feature,
    valid target endpoints are defined in InceptionI3d.VALID_ENDPOINTS
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    """
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video.transpose(1, 2),
                                          target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)

    return feat

def calculate_i3d_activations(video1, video2, i3d_model, device):
    """Calculate VFID metric.
        video1: list[PIL.Image]
        video2: list[PIL.Image]
    """
    video1 = to_tensors()(video1).unsqueeze(0).to(device)
    video2 = to_tensors()(video2).unsqueeze(0).to(device)
    video1_activations = get_i3d_activations(
        video1, i3d_model).cpu().numpy().flatten()
    video2_activations = get_i3d_activations(
        video2, i3d_model).cpu().numpy().flatten()

    return video1_activations, video2_activations

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +  # NOQA
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_vfid(real_activations, fake_activations):
    """
    Given two distribution of features, compute the FID score between them
    Params:
        real_activations: list[ndarray]
        fake_activations: list[ndarray]
    """
    m1 = np.mean(real_activations, axis=0)
    m2 = np.mean(fake_activations, axis=0)
    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(fake_activations, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)

class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(self,
                 in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,  # we always want padding to be 0 here. We will
            # dynamically pad based on input size in forward function
            bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels,
                                     eps=0.001,
                                     momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels,
                         output_channels=out_channels[0],
                         kernel_shape=[1, 1, 1],
                         padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels,
                          output_channels=out_channels[1],
                          kernel_shape=[1, 1, 1],
                          padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1],
                          output_channels=out_channels[2],
                          kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels,
                          output_channels=out_channels[3],
                          kernel_shape=[1, 1, 1],
                          padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3],
                          output_channels=out_channels[4],
                          kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1),
                                        padding=0)
        self.b3b = Unit3D(in_channels=in_channels,
                          output_channels=out_channels[5],
                          kernel_shape=[1, 1, 1],
                          padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self,
                 num_classes=400,
                 spatial_squeeze=True,
                 final_endpoint='Logits',
                 name='inception_i3d',
                 in_channels=3,
                 dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' %
                             self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels,
                                            output_channels=64,
                                            kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2),
                                            padding=(3, 3, 3),
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64,
                                            output_channels=64,
                                            kernel_shape=[1, 1, 1],
                                            padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64,
                                            output_channels=192,
                                            kernel_shape=[3, 3, 3],
                                            padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192,
                                                     [64, 96, 128, 16, 32, 32],
                                                     name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128,
                             output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128,
                             output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](
                    x)  # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits

    def extract_features(self, x, target_endpoint='Logits'):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
                if end_point == target_endpoint:
                    break
        if target_endpoint == 'Logits':
            return x.mean(4).mean(3).mean(2)
        else:
            return x

###########################
# Main metrics calculator class
###########################

class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device=device
        self.clip_metric_calculator = CLIPScore(model_name_or_path="zer0int/LongCLIP-L-Diffusers").to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.l1_metric_calculator = MeanAbsoluteError().to(device)
                
        self.clip_model = clip.load("ViT-B/32", device=device)[0]
        
    def calculate_clip_similarity(self, img, txt):
        img = np.array(img).astype(np.uint8)
        img_tensor=torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        img_tensor = img_tensor.to(device=self.device, dtype=torch.uint8)
        
        self.clip_metric_calculator.reset()
        with torch.no_grad():
            score = self.clip_metric_calculator(img_tensor, [txt])
        
        return float(score.detach().cpu().item())
    
    def calculate_psnr(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_gt = img_gt * mask
            img_pred = img_pred * mask
            
        img_pred_tensor=torch.from_numpy(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.from_numpy(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)

        self.psnr_metric_calculator.reset()
        with torch.no_grad():    
            score = self.psnr_metric_calculator(img_pred_tensor, img_gt_tensor)
        
        return float(score.detach().cpu().item())
    
    def calculate_lpips(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_gt = img_gt * mask
            img_pred = img_pred * mask
            
        img_pred_tensor=torch.from_numpy(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.from_numpy(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
        
        self.lpips_metric_calculator.reset()
        with torch.no_grad():
            score =  self.lpips_metric_calculator(img_pred_tensor*2-1, img_gt_tensor*2-1)
        
        return float(score.detach().cpu().item())
    
    def calculate_mse(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
            
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_gt = img_gt * mask
            img_pred = img_pred * mask
            
        img_pred_tensor=torch.from_numpy(img_pred).permute(2,0,1).to(self.device)
        img_gt_tensor=torch.from_numpy(img_gt).permute(2,0,1).to(self.device)
        
        self.mse_metric_calculator.reset()
        with torch.no_grad():
            score =  self.mse_metric_calculator(img_pred_tensor.contiguous(),img_gt_tensor.contiguous())
        
        return float(score.detach().cpu().item())
    
    def calculate_mae(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_gt = img_gt * mask
            img_pred = img_pred * mask
            
        img_pred_tensor = torch.from_numpy(img_pred).permute(2,0,1).to(self.device)
        img_gt_tensor = torch.from_numpy(img_gt).permute(2,0,1).to(self.device)

        self.l1_metric_calculator.reset()
        with torch.no_grad():    
            score = self.l1_metric_calculator(img_pred_tensor.contiguous(), img_gt_tensor.contiguous())
        
        return float(score.detach().cpu().item())

    def calculate_ssim(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_gt = img_gt * mask
            img_pred = img_pred * mask
            
        img_pred_tensor=torch.from_numpy(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.from_numpy(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)

        self.ssim_metric_calculator.reset()
        with torch.no_grad():    
            score =  self.ssim_metric_calculator(img_pred_tensor,img_gt_tensor)
        
        return float(score.detach().cpu().item())
    
    def calculate_temporal_consistency(self, images, masks=None):
        """Calculate temporal consistency between video frames, supports masked region computation.
        
        Args:
            images: Input video frames as numpy array or tensor with shape [N,H,W,C], 
                value range [0,255]
            masks: Optional mask tensor/numpy array with shape [N,H,W,1], 
                value range [0,1] where 1 indicates regions to compute
                
        Returns:
            float: Average temporal consistency score
        """
        # 确保输入是tensor并且在正确的设备上
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(self.device)
        
        # Normalize to [0,1] range
        if images.max() > 1.0:
            images = images / 255.0
            
        # Convert to CLIP expected format
        if images.shape[-1] == 3:  # If in [N,H,W,3] format
            images = images.permute(0,3,1,2)  # Convert to [N,3,H,W]

        # Mask processing if provided
        if masks is not None:
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks).float()
            masks = masks.to(self.device)
            
            # Ensure proper mask dimensions
            if masks.ndim == 3:  # [N,H,W]
                masks = masks.unsqueeze(-1)
            if masks.shape[-1] == 1 or masks.shape[-1] == 3:  # If in [N,H,W,C] format
                masks = masks.permute(0,3,1,2)  # Convert to [N,C,H,W]
            # Apply mask to images
            images = images * masks
        
        # Resize images to CLIP expected 224x224
        B, C, H, W = images.shape
        images = images.reshape(-1, C, H, W)  # Flatten batch dimension
        resize = transforms.Resize((224, 224), antialias=True)
        images = resize(images)
        images = images.reshape(B, C, 224, 224)  # Restore batch dimension

        # CLIP normalization (required for correct embeddings)
        clip_mean = torch.tensor([0.48145466, 0.45782750, 0.40821073], device=self.device).reshape(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).reshape(1, 3, 1, 1)
        if C == 3:
            images = (images - clip_mean) / clip_std
            
        # Extract and normalize features
        self.clip_model.eval()
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            normalized_features = torch.nn.functional.normalize(image_features, dim=1)

        # Cosine similarity between consecutive frames
        sims = (normalized_features[:-1] * normalized_features[1:]).sum(dim=1)  # (B-1,)
        return float(sims.mean().item())
    