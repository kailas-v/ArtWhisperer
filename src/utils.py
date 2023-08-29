import datetime
from hashlib import sha256
import os
import numpy as np
import pickle
import PIL
import PIL.Image
import pytz
from pytz import timezone
import shutil
from src import constants

# -- Datetime --
def get_now():
    today = datetime.datetime.now(tz=pytz.utc)
    today_PST = today.astimezone(timezone('US/Pacific'))
    return today_PST
def get_today():
    return get_now().date()
def get_timestamp():
    return get_now().strftime('%Y-%m-%d, %H:%M:%S (PST)')
def get_date():
    return get_now().strftime('%Y-%m-%d')
def str_to_date(s):
    return datetime.datetime.strptime(s, '%Y-%m-%d').date()
    

# -- Pickle --
def load_pkl(path, default=None):
    try:
        if path[-4:] != ".pkl":
            path = f"{path}.pkl"
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return default
def save_pkl(data, path):
    try:
        if path[-4:] != ".pkl":
            path = f"{path}.pkl"
        path_parent = "/".join(path.split("/")[:-1])
        os.makedirs(path_parent, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except:
        return False

# -- Seed from object --
def rng_from_obj(obj):
    if isinstance(obj, (str,)):
        hash_ = sha256(obj.encode())
        seed = np.frombuffer(hash_.digest(), dtype='uint32')
    elif isinstance(obj, (int,)):
        seed = obj
    else:
        hash_ = sha256(obj)
        seed = np.frombuffer(hash_.digest(), dtype='uint32')
    return np.random.RandomState(seed)

def str_hash(s):
    return sha256(s.encode("utf-8")).hexdigest()

def strs_hash(strings):
    return str_hash("".join([str_hash(s) for s in strings]))

# -- Directories --
def listdir_ext(path, exts):
    return  [f for f in os.listdir(path) if f[-3:].lower() in exts]


# -- Images --
def load_image(path, as_PIL=False):
    image = PIL.Image.open(str(path))
    if not as_PIL:
        image = np.array(image)
    return image


# -- Numpy --
def nan_array(size):
    arr = np.zeros(size)
    arr[:] = np.nan
    return arr

def pad_last(array, n_pad, default_pad=np.nan):
    pad_value = np.nan
    if np.size(array) > 0:
        pad_value = array[-1]
    return np.concatenate((array, np.zeros(n_pad)+pad_value))

# -- Experiment related --
LEN_SHORT_ID = 15
S = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,."
def hex_to_url(s):
    build_str = []
    for i in range(0, len(s), 3):
        si = int(s[i:i+3], 16)
        si1 = si // 64
        si2 = si % 64
        build_str.append(f"{S[si1]}{S[si2]}")
    return "".join(build_str)
def shorten_user_id(user_id):
    try:
        return hex_to_url(user_id[:LEN_SHORT_ID])
    except:
        return user_id

def load_target_image(target_num):
    image_path = f"{constants.TARGET_IMAGE_PATH}/{target_num:03d}.jpg"
    return load_image(image_path)
def load_target_meta(target_num):
    meta_path = f"{constants.TARGET_META_PATH}/{target_num:03d}.npy"
    return np.load(meta_path, allow_pickle=True).item()



# -- Text file --
def load_text_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return liens