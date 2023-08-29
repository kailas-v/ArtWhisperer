import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import imageio
import numpy as np
from src.im_utils import load_image
from PIL import Image, ImageDraw

def best_fit_line(X, Y):
    X,Y = np.array(X), np.array(Y)
    x_ = np.linspace(np.min(X), np.max(X), num=100)
    X = np.concatenate((np.ones((1,len(X))), X[None,:]), axis=0)
    b,m = np.linalg.inv(X@X.T) @ X @ Y

    
    y_ = b + m*x_
    return (x_,y_), (m,b)

def avg_bin(X, Y, bins=10):
    X,Y = np.array(X), np.array(Y)
    _,bin_edges = np.histogram(X, bins=bins)
    bin_edges = list(bin_edges)
    x_,y_ = [], []
    for s,e in zip(bin_edges[:-1], bin_edges[1:]):
        x_.append((s+e)/2)
        y_.append(np.mean(Y[(X>=s) & (X < e)]))
    return (x_, y_)


    X = np.concatenate((np.ones((1,len(X))), X[None,:]), axis=0)
    b,m = np.linalg.inv(X@X.T) @ X @ Y

    x_ = np.linspace(np.min(X), np.max(X), num=100)
    y_ = b + m*x_
    return (x_,y_), (m,b)

PLOT_COLORS = list(mcolors.TABLEAU_COLORS.values()) + ['limegreen']

def new_fig(nrows=1,ncols=1,figsize=(12,8),**kwargs):
    plt.rcParams.update({'font.size':14})
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize,**kwargs)
    return fig,axs
def setup_axes(axs, no_legend=False):
    for axi in axs:
        axi.grid()
        if not no_legend:
            axi.legend()


def crop_whitespace(image):
    # Convert image to grayscale
    image_gray = image.convert('L')

    # Convert image to NumPy array
    image_array = np.array(image_gray)

    # Find non-zero indices along the rows and columns
    rows = np.where(np.any(image_array != 255, axis=1))[0]
    cols = np.where(np.any(image_array != 255, axis=0))[0]

    # Crop the image
    cropped_image = image.crop((cols.min(), rows.min(), cols.max()+1, rows.max()+1))

    return cropped_image

def add_rectangle(image, rect=None, alpha=0.2, color=(255,0,0), all=False, threshold=0, skip_rects=[]):
    image = image.convert("RGBA")
    # Open the image
    width, height = image.size

    # Create a blank overlay image with transparent background
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if rect is None:
        rect = [[0,height], [0,width]]
    if rect[0][0] is None:
        rect[0][0] = 0
    if rect[0][1] is None:
        rect[0][1] = height
    if rect[1][0] is None:
        rect[1][0] = 0
    if rect[1][1] is None:
        rect[1][1] = width

    # Iterate over each pixel in the image
    for y in range(rect[0][0], rect[0][1]):
        for x in range(rect[1][0], rect[1][1]):
            skip = False
            for sr in skip_rects:
                if (y <= sr[0][1]) and (y > sr[0][0]) and (x <= sr[1][1]) and (x > sr[1][0]):
                    skip = True
                    break
            if skip:
                continue
            # Get the pixel value at (x, y)
            pixel = image.getpixel((x, y))

            # Check if the pixel is white
            pixel_diff = np.sum(np.abs(np.array(pixel[:3]) - np.array((255,255,255))))
            if all or (pixel_diff <= threshold):
                # Calculate the new pixel value with alpha
                new_pixel = (color[0], color[1], color[2], int(alpha * 255))

                # Draw a rectangle at the pixel location on the overlay image
                draw.rectangle([(x, y), (x+1, y+1)], fill=new_pixel)

    # Composite the overlay image onto the original image
    result = Image.alpha_composite(image, overlay)

    # Save or display the result
    return result.convert("RGB")
    
def save_with_cropped_whitespace(src_path, dst_path=None):
    if dst_path is None:
        dst_path = src_path
    img = load_image(src_path, as_PIL=True)
    img = crop_whitespace(img)
    img.save(dst_path)

def save_with_colored_rectangle(src_path, dst_path=None, rect=None, color=(255,0,0), alpha=0.2, all=False, threshold=0, skip_rects=[]):
    if dst_path is None:
        dst_path = src_path
    img = load_image(src_path, as_PIL=True)
    img = add_rectangle(img, rect=rect, alpha=alpha, color=color, all=all, threshold=threshold, skip_rects=skip_rects)
    img.save(dst_path)

def save_fig(filename, path, exts=['png', 'pdf'], fig=None, save=True, show=False, close=True, tight=True):
    if fig is None: fig = plt.gcf()
    if path is not None: os.makedirs(path, exist_ok=True)
    save_paths = []
    for ext in exts:
        save_paths.append(os.path.join(path, f"{filename}.{ext}"))
        if save:
            if tight:
                fig.savefig(save_paths[-1], bbox_inches='tight', transparent=True)
            else:
                fig.savefig(save_paths[-1], transparent=True)
    if show: plt.show()
    if close: plt.close()
    return save_paths

def save_gif(images, filename, path):
    if path is not None: os.makedirs(path, exist_ok=True)
    gif_path = os.path.join(path, filename)
    imageio.mimsave(gif_path, images)
    return gif_path
