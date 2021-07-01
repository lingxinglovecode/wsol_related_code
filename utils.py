import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import matplotlib.ticker as ticker
def normalize_tensor(x):
    #输入为[B,C,H,W]
    channel_vector = x.view(x.size()[0], x.size()[1], -1)
    minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
    maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
    normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
    normalized_tensor = normalized_vector.view(x.size())
    return normalized_tensor
def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam

def save_tight(filepath):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)

    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)


def apply_heatmap(image, heatmap, alpha=0.6, display=False, save=None, cmap='viridis', axis='on', verbose=False):

    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heatmap, (height, width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    # print('Min {} Max {}'.format(min_value,max_value))
    normalized_heat_map =1- (heat_map_resized - min_value) / (max_value - min_value)

    # display
    plt.imshow(image)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        if verbose:
            print('save image: ' + save)
        # plt.savefig(save, bbox_inches='tight', pad_inches=0)
        save_tight(save)

    return plt

def save_heatmap(heatmap,save=None,max_frame_size=224):
    heat_map_resized = transform.resize(heatmap, (max_frame_size, max_frame_size))
    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = 255* (heat_map_resized - min_value) / (max_value - min_value)
    imageio.imwrite(save,normalized_heat_map)