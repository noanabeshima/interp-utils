from PIL import Image
import torch
import numpy as np
from scipy import stats

import math
import einops
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt

def is_tensor(x):
    return isinstance(x, torch.Tensor) or isinstance(x, torch.nn.parameter.Parameter)

def tensor_to_numpy(x):
    # if x is a tensor, convert to numpy array
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, torch.nn.parameter.Parameter):
        x = x.detach().cpu().numpy()
    return x

def render_array(arr, scale: int = 1, raw_array=False):
    arr = tensor_to_numpy(arr)
    assert isinstance(arr, np.ndarray)
    arr = np.squeeze(arr)
    assert len(arr.shape) == 2
    if scale != 1:
        arr = np.kron(arr, np.ones((scale, scale)))

    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() < 0.0 or arr.max() > 1.0:
            if arr.min() > -1e-2 and arr.max() < 1.0 + 1e-2:
                arr = np.clip(arr, 0, 1)
            else:
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-2)
        arr = (255 * arr // 1).astype(np.uint8)
    else:
        assert np.issubdtype(arr.dtype, np.integer)

    if raw_array is True:
        return arr
    else:
        return Image.fromarray(arr)


def render_array_w_sign(arr, scale: int = 1):
    pos = np.clip(arr, 0, None)
    neg = np.clip(arr, None, 0).abs()
    pos = render_array(pos / arr.abs().max(), scale=scale, raw_array=True)
    neg = render_array(neg / arr.abs().max(), scale=scale, raw_array=True)

    both = np.zeros(pos.shape + (3,), dtype=np.uint8)
    both[..., 0] = neg
    both[..., 1] = pos

    return Image.fromarray(both)


def str_arr_add(*args):
    """
    Casts tensors/nparrays to numpy string arrays and adds all items together,
    casting to string and broadcasting if necessary
    """
    if len(args) == 0:
        return ""
    args = list(args)
    for i, item in enumerate(args):
        item = tensor_to_numpy(item)
        if isinstance(item, np.ndarray):
            args[i] = item.astype(str)
    res = args[0]
    for item in args[1:]:
        res = np.core.defchararray.add(res, item)
    return res


def heatmap(
    arr,
    perm_0=None,
    perm_1=None,
    dim_names=("row", "col"),
    info_0=None,
    info_1=None,
    include_idx=(True, True),
    title=None,
    mask_0=None,
    mask_1=None,
    sort_0=None,
    sort_1=None
):
    """
    arr: 2d numpy or torch array to render
    dim_names : (str, str), names of dim 0 and dim 1 respectively
    info_0, info_1 : dictionary of string keys to list of strings describing the indices of dim 0 and dim 1 respectively

    sort_0, sort_1: 1d arrays of indices to sort the rows and columns of the heatmap by
    """


    assert not(perm_0 is not None and sort_0 is not None), "Cannot provide both perm_0 and sort_0"
    assert not(perm_1 is not None and sort_1 is not None), "Cannot provide both perm_1 and sort_1"

    # convert arr to numpy array
    arr = tensor_to_numpy(arr)
    assert isinstance(arr, np.ndarray)

    # Create default title if none is provided
    if title is None:
        if dim_names == ("row", "col"):
            title = f"{arr.shape}"
        else:
            title = f"({dim_names[0]}, {dim_names[1]})"
    
    # get permutations from sort arrays
    if sort_0 is not None:
        assert len(sort_0.shape) == 1
        perm_0 = torch.tensor(tensor_to_numpy(sort_0)).topk(k=len(sort_0)).indices
    if sort_1 is not None:
        assert len(sort_1.shape) == 1
        perm_1 = torch.tensor(tensor_to_numpy(sort_1)).topk(k=len(sort_1)).indices
    
    # if permutations are not provided, use the identity permutation
    perm_0 = np.arange(arr.shape[0]) if perm_0 is None else tensor_to_numpy(perm_0)
    perm_1 = np.arange(arr.shape[1]) if perm_1 is None else tensor_to_numpy(perm_1)

    def construct_dim_info(dim_info: dict, dim_name: str, dim_len, perm, mask=None, include_idx=False):
        dim_info = {} if dim_info is None else dim_info

        if include_idx is True:
            dim_info[f"{dim_name}"] = np.arange(dim_len)

        for k, v in dim_info.items():
            if is_tensor(v):
                dim_info[k] = tensor_to_numpy(v)
            else:
                if not isinstance(v, np.ndarray):
                    dim_info[k] = np.array(v)

        dim_info = {k: v[perm] for k, v in dim_info.items()}
        
        if mask is not None:
            mask = tensor_to_numpy(mask)[perm]
            for k, v in dim_info.items():
                dim_info[k] = v[mask]
    
        dim_info = str_arr_add(
            *[str_arr_add(k + ": ", v, "<br>") for k, v in dim_info.items()]
        )
        dim_info = np.array(dim_info).astype(str).tolist()
        return dim_info

    # Construct hovertemplate and dim info for each dimension (0 and 1)
    hovertemplate = ""
    if info_0 is not None or include_idx[0]:
        hovertemplate += "%{y}"
        info_0 = construct_dim_info(info_0, dim_names[0], arr.shape[0], perm_0, mask_0, include_idx[0])
    if info_1 is not None or include_idx[1]:
        hovertemplate += "%{x}"
        info_1 = construct_dim_info(info_1, dim_names[1], arr.shape[1], perm_1, mask_1, include_idx[1])
    hovertemplate += "val: %{z:.2f}<extra></extra>"

    # apply masks and permutations
    arr = arr[perm_0][:,perm_1]
    if mask_0 is not None:
        arr = arr[mask_0[perm_0]]
    if mask_1 is not None:
        arr = arr[:,mask_1[perm_1]]

    # Create the plotly.graph_objects figure
    layout = go.Layout(yaxis=dict(autorange="reversed"))

    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            y=info_0,
            x=info_1,
            hovertemplate=hovertemplate,
            colorscale="Viridis",
        ),
        layout=layout,
    )

    fig.update_layout(
        xaxis_title=f"{dim_names[1]} ({arr.shape[1]})",
        yaxis_title=f"{dim_names[0]} ({arr.shape[0]})",
        title=title,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def qq_plot(x, dist="norm", sparams=(), hovertext=None):
    x = x.squeeze()
    assert len(x.shape) == 1
    perm = x.topk(x.shape[0], largest=False).indices
    hovertext = np.array(hovertext)[perm] if hovertext is not None else None
    qq = stats.probplot(x[perm], dist=dist, sparams=sparams)
    x = np.array([qq[0][0][0], qq[0][0][-1]])
    fig = go.Figure()
    fig.add_scatter(x=qq[0][0], y=qq[0][1], mode="markers", hovertext=hovertext)
    fig.update_xaxes(title="theoretical quantiles")
    fig.update_yaxes(title="actual quantiles")

    fig.add_scatter(x=x, y=qq[1][1] + qq[1][0] * x, mode="lines")
    fig.layout.update(showlegend=False)
    fig.show()


def get_image_grid(images, width: int = -1, scale: int = 1):
    """images is a list of PIL.Image images"""

    assert scale >= 1

    def find_closest_factors(n):
        for i in range(int(np.sqrt(n).item()) + 1, 0, -1):
            if n % i == 0:
                return i, n // i

    image_array = np.stack([np.array(image) for image in images], axis=0)
    if width == -1:
        rows, columns = find_closest_factors(len(images))
    else:
        assert isinstance(width, int)
        rows = math.ceil(len(images) / width)
        columns = width
        empty_images_needed = rows * columns - len(images)

        # add the empty images
        image_array = np.concatenate(
            [
                image_array,
                np.zeros((empty_images_needed,) + image_array.shape[1:]).astype(
                    np.uint8
                ),
            ],
            axis=0,
        )

    image_arr = einops.rearrange(
        image_array, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=rows, b2=columns
    )
    if scale != 1:
        image_arr = np.kron(image_arr, np.ones((scale, scale, 1))).astype(np.uint8)
    image_grid = Image.fromarray(image_arr)
    return image_grid


def plthist(x, *args, **kwargs):
    x = tensor_to_numpy(x)
    plt.histogram(x, *args, **kwargs)
    plt.show()


def hist(x, *args, **kwargs):
    x = tensor_to_numpy(x)
    return px.histogram(x, *args, **kwargs)

