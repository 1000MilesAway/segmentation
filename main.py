from np_process import preprocess, postprocess
from torch_process import postprocess as torch_post
import onnxruntime as ort
import numpy as np
import cv2
import torch


def denorm(persons, w, h):
    for person in persons:
        for point in person:
            point[0] *= w
            point[1] *= h


image = cv2.imread('que.jpg')
h, w, _ = image.shape
input_data = preprocess(image, 1440, 800)
input_data = np.expand_dims(input_data.astype(np.float32), axis=0)
ort_sess = ort.InferenceSession('model.onnx')
outputs = ort_sess.run(None, {'input': input_data})

jija = torch.jit.load('scriptmodule.pt')
torch_data = []
for o in outputs:
    torch_data.append(torch.Tensor(o))
bl = jija(*torch_data) # , torch.Tensor([1440, 800])


result = postprocess(outputs)
result_torch = torch_post(outputs, torch.Tensor([1440, 800]))

denorm(result, 1440, 800)
denorm(result_torch, 1440, 800)
print(outputs)
