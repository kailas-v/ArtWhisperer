import PIL
import PIL.Image
import base64
import io
import skimage
import numpy as np

def encode_np_image(np_image, filetype='PNG'):
    img = PIL.Image.fromarray(np_image.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    img.save(file_object, filetype)
    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)
    return file_object

def decode_base64_to_image(encoding):
    content = encoding.split(';')[1]
    image_encoded = content.split(',')[1]
    return PIL.Image.open(io.BytesIO(base64.b64decode(image_encoded)))

def encode_file_to_base64(f, type="image", ext=None, header=True):
    with open(f, "rb") as file:
        encoded_string = base64.b64encode(file.read())
        base64_str = str(encoded_string, 'utf-8')
        if not header:
            return base64_str
        if ext is None:
            ext = f.split(".")[-1]
        return "data:" + type + "/" + ext + ";base64," + base64_str

def encode_array_to_base64(image_array, filetype='PNG'):
    if filetype=='JPG':
        filetype = 'JPEG'
    with io.BytesIO() as output_bytes:
        PIL_image = PIL.Image.fromarray(skimage.img_as_ubyte(image_array))
        if filetype=='JPEG':
           PIL_image.save(output_bytes, filetype, quality=20, optimize=True, progressive=True)
        else:
            PIL_image.save(output_bytes, filetype)
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return f"data:image/{filetype.lower()};base64," + base64_str


def load_image(path, as_PIL=False):
    image = PIL.Image.open(str(path))
    if not as_PIL:
        image = np.array(image)
    return image

def resize_and_crop_512_square(image):
    if image.width < image.height:
        resized_image = image.resize((512, int(512 * image.height / image.width)))
    else:
        resized_image = image.resize((int(512 * image.width / image.height), 512))

    # Crop the center of the image to 512x512
    left = (resized_image.width - 512) / 2
    top = (resized_image.height - 512) / 2
    right = (resized_image.width + 512) / 2
    bottom = (resized_image.height + 512) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    # Return the cropped and resized image to a file
    return cropped_image