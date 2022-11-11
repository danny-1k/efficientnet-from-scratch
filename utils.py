from PIL import Image


def calculate_factors(v='b0', alpha=1.2, beta=1.1):
    # gamma value from paper does not give the ouput channels from the paper

    if v == 'b0':
        return 1,1

    phis = {
        'b1':.5,
        'b2':1,
        'b3':2,
        'b4':3,
        'b5':4,
        'b6':5,
        'b7':6,
        'b8':7
    }

    phi = phis[v]

    depth_factor = alpha**phi
    width_factor = beta**phi

    return depth_factor, width_factor

def get_image_size(v='b0'):

    sizes = {
        'b0':224,
        'b1':240,
        'b2':260,
        'b3':300,
        'b4':380,
        'b5':456,
        'b6':528,
        'b7':600,
        'b8':672,
    }

    return sizes[v]


def get_dropout(v):
    values = {
        'b0':0.2,
        'b1':0.2,
        'b2':0.3,
        'b3':0.3,
        'b4':0.4,
        'b5':0.4,
        'b6':0.5,
        'b7':0.5,
        'b8':0.5,
    }

    return values[v]


def read_image(path):
    img = Image.open(path)
    return img
