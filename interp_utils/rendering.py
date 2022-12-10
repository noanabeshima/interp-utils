from PIL import Image
import torch
import numpy as np
from scipy import stats

import math
import einops
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt



def render_array(arr, scale: int = 1, raw_array=False):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, torch.nn.parameter.Parameter):
        arr = arr.detach().cpu().numpy()
    assert isinstance(arr, np.ndarray)
    arr = np.squeeze(arr)
    assert len(arr.shape) == 2
    if scale != 1:
        arr = np.kron(arr, np.ones((scale,scale)))
    
    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() < 0.0 or arr.max() > 1.0:
            if arr.min() > -1e-2 and arr.max() < 1.0+1e-2:
                arr = np.clip(arr, 0, 1)
            else:
                arr = (arr - arr.min())/(arr.max()-arr.min()+1e-2)
        arr = (255*arr // 1).astype(np.uint8)
    else:
        assert np.issubdtype(arr.dtype, np.integer)

    if raw_array is True:
        return arr
    else:
        return Image.fromarray(arr)

def render_array_w_sign(arr, scale: int = 1):
    pos = np.clip(arr, 0, None)
    neg = np.clip(arr, None, 0).abs()
    pos = render_array(pos/arr.abs().max(), scale=scale, raw_array=True)
    neg = render_array(neg/arr.abs().max(), scale=scale, raw_array=True)

    both = np.zeros(pos.shape+(3,), dtype=np.uint8)
    both[...,0] = neg
    both[...,1] = pos

    return Image.fromarray(both)

def str_arr_add(*args):
    '''
    Casts tensors/nparrays to numpy string arrays and adds all items together,
    casting to string and broadcasting if necessary
    '''
    if len(args) == 0:
        return ''
    args = list(args)
    for i, item in enumerate(args):
        if isinstance(item, torch.Tensor):
            item = item.numpy()
        if isinstance(item, np.ndarray):
            args[i] = item.astype(str)
    res = args[0]
    for item in args[1:]:
        res = np.core.defchararray.add(res, item)
    return res


def heatmap(arr, perm_0=False, perm_1=False, dim_names = ('row', 'col'), info_0=None, info_1=None, include_idx=(True, True), title=False):
    '''
    name_0, name_1 : names of dim 0 and dim 1 respectively
    info_0, info_1 : dictionary of string keys to list of strings describing the indices of dim 0 and dim 1 respectively

    include_idx[i] must be true if dim_i_info is False
    '''
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    assert isinstance(arr, np.ndarray)
    # assert not (info_0 is False and include_idx[0] is False)
    # assert not (info_1 is False and include_idx[1] is False)

    if title is False:
        if dim_names == ('row', 'col'):
            title = f'{arr.shape}'
        else:
            title = f'({dim_names[0]}, {dim_names[1]})'

    perm_0 = np.arange(arr.shape[0]) if perm_0 is False else perm_0
    perm_1 = np.arange(arr.shape[1]) if perm_1 is False else perm_1


    if info_0 is None and include_idx[0] is True:
        info_0 = {} 
    if info_1 is None and include_idx[1] is True:
        info_1 = {}
    
    if info_0 is not None and include_idx[0]:
        info_0[f'{dim_names[0]}'] = np.arange(arr.shape[0])
    if info_1 is not None and include_idx[1]:
        info_1[f'{dim_names[1]}'] = np.arange(arr.shape[1])

    assert info_0 != {}
    assert info_1 != {}

    hovertemplate = ''
    if info_0 is not None:
        hovertemplate += '%{y}'
        info_0 = {k: np.array(v)[perm_0] for k, v in info_0.items()}
        info_0 = str_arr_add(*[str_arr_add(k+': ', v, '<br>') for k, v in info_0.items()])
        info_0 = np.array(info_0).astype(str).tolist()

    if info_1 is not None:
        hovertemplate += '%{x}'
        info_1 = {k: np.array(v)[perm_1] for k, v in info_1.items()}
        info_1 = str_arr_add(*[str_arr_add(k+': ', v, '<br>') for k, v in info_1.items()])
        info_1 = np.array(info_1).astype(str).tolist()
    hovertemplate += 'val: %{z:.2f}<extra></extra>'    

    layout = go.Layout(
        yaxis=dict(autorange='reversed')
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=arr[perm_0][:,perm_1],
            x=info_1,
            y=info_0,
            hovertemplate=hovertemplate,
            colorscale='Viridis',
        ),
        layout=layout)

    fig.update_layout(xaxis_title=f"{dim_names[1]} ({arr.shape[1]})", yaxis_title=f"{dim_names[0]} ({arr.shape[0]})", title=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def qq_plot(x, dist='norm', sparams=(), hovertext=None):
    x = x.squeeze()
    assert len(x.shape) == 1
    perm = x.topk(x.shape[0], largest=False).indices
    hovertext = np.array(hovertext)[perm] if hovertext is not None else None
    qq = stats.probplot(x[perm], dist='norm', sparams=sparams)
    x = np.array([qq[0][0][0], qq[0][0][-1]])
    fig = go.Figure()
    fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers', hovertext=hovertext)
    fig.update_xaxes(title='theoretical quantiles')
    fig.update_yaxes(title='actual quantiles')

    fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')
    fig.layout.update(showlegend=False)
    fig.show()


def get_image_grid(images, width: int = -1, scale: int = 1):
    '''images is a list of PIL.Image images'''

    assert scale >= 1

    def find_closest_factors(n):
        for i in range(int(np.sqrt(n).item())+1, 0, -1):
            if n % i == 0:
                return i, n//i

    image_array = np.stack([np.array(image) for image in images], axis=0)
    if width == -1:
        rows, columns = find_closest_factors(len(images))
    else:
        assert isinstance(width, int)
        rows = math.ceil(len(images) / width)
        columns = width
        empty_images_needed = rows * columns - len(images)
        
        # add the empty images
        image_array = np.concatenate([image_array, np.zeros((empty_images_needed,)+image_array.shape[1:]).astype(np.uint8)], axis=0)

    image_arr = einops.rearrange(image_array, '(b1 b2) h w c -> (b1 h) (b2 w) c', b1=rows, b2=columns)
    if scale != 1:
        image_arr = np.kron(image_arr, np.ones((scale,scale,1))).astype(np.uint8)
    image_grid = Image.fromarray(image_arr)
    return image_grid

def preprocess_arr(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x

def plthist(x, *args, **kwargs):
    x = preprocess_arr(x)
    plt.histogram(x, *args, **kwargs)
    plt.show()

hist = px.histogram