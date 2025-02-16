import sys
import json
import requests
import numpy as np
from PIL import Image
from pathlib import Path


def resize(img, new_size):
    img = Image.fromarray(img)
    img = img.resize(new_size)
    return np.array(img)

def prepare_digit(img):
    img = resize(img, (28, 28))
    img = img.astype(np.float32)/255
    if len(img.shape) > 2:
        img = np.mean(img, axis=2)
    img = (1. - img).astype(np.float32)
    img = np.reshape(img, (28, 28, 1))
    return img

def prepare_input(img):
    img = img.convert("F")
    img = np.array(img, dtype=np.uint8)
    img = prepare_digit(img)
    return img[:,:,0][None,:,:]

def decode_predictions(data):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    preds = np.argmax(data, axis=-1)
    preds = [class_names[i] for i in preds]

    return preds

img_fpath = Path(sys.argv[1]).resolve()
print('File:', img_fpath)
img = Image.open(img_fpath)

inp = prepare_input(img)

request_data = json.dumps({
    "instances": inp.tolist()
})
headers = {"content-type": "application/json"}

json_response = requests.post(
    'http://localhost:8502/models/my_model/:predict',
    data=request_data, headers=headers
)

response = json.loads(json_response.content.decode())

if 'prdictions' in response:
    preds = decode_predictions(response['prdictions'])
    print('Prediction:', preds[0])
else:
    print(response)