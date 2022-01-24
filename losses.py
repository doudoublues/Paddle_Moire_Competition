import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
import math



def Sobel_loss(output, target):
    b, c, h, w = output.shape
    filter_x  = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    filter_x = np.expand_dims(filter_x, 0)
    filter_x = np.tile(np.expand_dims(filter_x, 0), [c, 1, 1, 1])
    filter_x = paddle.to_tensor(filter_x)

    filter_y = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    filter_y = np.expand_dims(filter_y, 0)
    filter_y = np.tile(np.expand_dims(filter_y, 0), [c, 1, 1, 1])
    filter_y = paddle.to_tensor(filter_y)

    filter_a = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)
    filter_a = np.expand_dims(filter_a, 0)
    filter_a = np.tile(np.expand_dims(filter_a, 0), [c, 1, 1, 1])
    filter_a = paddle.to_tensor(filter_a)

    filter_b = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32)
    filter_b = np.expand_dims(filter_b, 0)
    filter_b = np.tile(np.expand_dims(filter_b, 0), [c, 1, 1, 1])
    filter_b = paddle.to_tensor(filter_b)


    output_gradient_x = paddle.square(F.conv2d(output, filter_x, groups=c, stride = 1, padding = 'SAME'))
    output_gradient_y = paddle.square(F.conv2d(output, filter_y, groups=c, stride = 1, padding = 'SAME'))
    output_gradient_a = paddle.square(F.conv2d(output, filter_a, groups=c, stride = 1, padding = 'SAME'))
    output_gradient_b = paddle.square(F.conv2d(output, filter_b, groups=c, stride = 1, padding = 'SAME'))

    output_gradients = paddle.sqrt(output_gradient_x + output_gradient_y +  output_gradient_a +  output_gradient_b + 1e-6)

    target_gradient_x = paddle.square(F.conv2d(target, filter_x, groups=c, stride = 1, padding = 'SAME'))
    target_gradient_y = paddle.square(F.conv2d(target, filter_y, groups=c, stride = 1, padding = 'SAME'))
    target_gradient_a = paddle.square(F.conv2d(target, filter_a, groups=c, stride = 1, padding = 'SAME'))
    target_gradient_b = paddle.square(F.conv2d(target, filter_b, groups=c, stride = 1, padding = 'SAME'))

    target_gradients = paddle.sqrt(target_gradient_x + target_gradient_y + target_gradient_a + target_gradient_b + 1e-6)

    mse = nn.MSELoss()

    loss = mse(output_gradients, target_gradients)
    
    return loss

class L1_Charbonnier_loss(nn.Layer):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = paddle.add(X, -Y)
        error = paddle.sqrt(diff * diff + self.eps)
        loss = paddle.mean(error)
        return loss


def create_window(window_size: int, sigma: float, channel: int):
    """
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    """
    coords = paddle.arange(window_size, dtype=paddle.float32)
    coords -= window_size // 2

    g = paddle.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape((1, 1, 1, -1)).tile((channel, 1, 1, 1))
    return g


def _gaussian_filter(x, window_1d, use_padding: bool):
    """
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    """

    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose((0, 1, 3, 2)), stride=1, padding=(padding, 0), groups=C)
    return out


def ssim(X, Y, window, data_range: float, use_padding: bool = False):
    """
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    """

    K1 = 0.1
    K2 = 0.3
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # Fixed the issue that the negative value of cs_map caused ms_ssim to output Nan.
    cs_map = F.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(axis=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(axis=(1, 2, 3))

    return ssim_val, cs


def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False, eps: float = 1e-8):
    """
    interface of ms-ssim
    :param X: a batch of images, (N,C,H,W)
    :param Y: a batch of images, (N,C,H,W)
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param weights: weights for different levels
    :param use_padding: padding image before conv
    :param eps: use for avoid grad nan.
    :return:
    """
    levels = weights.shape[0]
    weights = weights.reshape((levels, 1))  # weights[:, None]

    vals = []

    for i in range(levels):
        ss, cs = ssim(X, Y, window=window, data_range=data_range, use_padding=use_padding)
        multi_not_least = X.shape[-1] >= np.power(2, levels)
        if i < levels - 1 and multi_not_least:
            vals.append(cs)
            X = F.avg_pool2d(X, kernel_size=2, stride=2, ceil_mode=True)
            Y = F.avg_pool2d(Y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            vals.append(ss)

    vals = paddle.stack(vals, axis=0)
    # Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
    vals = vals.clip(min=eps)
    # The origin ms-ssim op.
    ms_ssim_val = paddle.prod(vals[:-1] ** weights[:-1] * vals[-1:] ** weights[-1:], axis=0)
    # The new ms-ssim op. But I don't know which is best.
    # ms_ssim_val = torch.prod(vals ** weights, dim=0)
    # In this file's image training demo. I feel the old ms-ssim more better. So I keep use old ms-ssim op.
    return ms_ssim_val


class MS_SSIM:
    __constants__ = ['data_range', 'use_padding', 'eps']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False, weights=None,
                 levels=None, eps=1e-8):
        """
        class for ms-ssim
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels
        :param use_padding: padding image before conv
        :param weights: weights for different levels. (default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        :param levels: number of downsampling
        :param eps: Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
        """

        assert window_size % 2 == 1, 'Window size must be odd.'
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

        self.window = create_window(window_size, window_sigma, channel)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.weights = paddle.to_tensor(weights, dtype=paddle.float32)

        if levels is not None:
            weights = weights[:levels]
            self.weights = weights / weights.sum()

    def __call__(self, X, Y):
        return (1 - ms_ssim(X, Y, window=self.window, data_range=self.data_range, weights=self.weights,
                       use_padding=self.use_padding, eps=self.eps)).mean()



class PSNR:
    """
    Peak signal-to-noise ratio metric
    """

    def __init__(self, max_val=1.0):
        self.max_val = max_val

    def __call__(self, expected, actual):
        """
        compute peak signal-to-noise ratio
        Args:
            real(Tensor): Actual result
            expected(Tensor): Expected result
        
        Returns:
            (float): Mean square error for input of expected and actual
        """
        mse_loss = nn.MSELoss(reduction="mean")
        mse = mse_loss(actual, expected).numpy()[0]

        if mse == 0:
            # prevent divided by zero
            mse = 0.000001
        psnr = 20 * math.log10(self.max_val / math.sqrt(mse))
        return psnr


if __name__ == '__main__':
    img1 = paddle.ones([5, 3, 224, 224])
    img2 = paddle.ones([5, 3, 224, 224])
    losser2 = MS_SSIM(data_range=1.)
    loss2 = losser2(img1, img2).mean()
    print(loss2)
    # Sobel_loss(img1,img2)
    # # net = LossNetwork()

    # out = net(img)

    