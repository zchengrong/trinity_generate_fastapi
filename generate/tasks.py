# tasks.py
import asyncio
import io
import random
import threading
import time
import uuid

import cv2
import mmcv
import numpy as np
import torch
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import tritonclient
from PIL import Image

from celery import Celery
from minio import Minio
from functools import partial

from tritonclient.utils import InferenceServerException
import torch.nn.functional as F

minio_client = Minio("www.minio.aida.com.hk:9000", access_key="vXKFLSJkYeEq2DrSZvkB", secret_key="uKTZT3x7C43WvPN9QTc99DiRkwddWZrG9Uh3JVlR", secure=True)
triton_client = grpcclient.InferenceServerClient(url="10.1.1.240:7001", verbose=False)
celery = Celery(
    "tasks",
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/1",

)

id_lock = threading.Lock()


def generate_uuid():
    with id_lock:
        unique_id = str(uuid.uuid1())
    return unique_id


def preprocess_image(image, category):
    height, width, _ = image.shape

    if category == "print" or category == "moodboard":
        square_size = min(height, width)
        start_x = (width - square_size) // 2
        start_y = (height - square_size) // 2
        cropped = image[start_y: start_y + square_size, start_x: start_x + square_size]
        resized_image = cv2.resize(cropped, (512, 512))

    elif category == "sketch":
        # below is the way that get "bigger" square image.
        max_dimension = max(height, width)
        square_image = np.ones((max_dimension, max_dimension, 3), dtype=np.uint8) * 255
        start_h = (max_dimension - height) // 2
        start_w = (max_dimension - width) // 2
        square_image[start_h:start_h + height, start_w:start_w + width] = image
        resized_image = cv2.resize(square_image, (512, 512))

    else:
        raise ValueError(f"wrong category {category}, only in moodboard, print and sketch!")

    return resized_image


def get_image(minio_client, category, image_url=None):
    try:
        response = minio_client.get_object(image_url.split('/')[0], image_url[image_url.find('/') + 1:])
        img = np.frombuffer(response.data, np.uint8)  # 转成8位无符号整型
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # 解码
        img = preprocess_image(img, category)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img = np.random.randn(512, 512, 3)
    return img


def get_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Edge = cv2.Canny(gray, 10, 150)
    kernel = np.ones((5, 5), np.uint8)
    Edge = cv2.dilate(Edge, kernel=kernel, iterations=1)
    Edge = cv2.erode(Edge, kernel=kernel, iterations=1)
    Contour, _ = cv2.findContours(Edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Contour = sorted(Contour, key=cv2.contourArea, reverse=True)
    return Contour


def get_mask(image_obj):
    pre_mask = None
    if len(image_obj.shape) == 2:
        image_obj = cv2.cvtColor(image_obj, cv2.COLOR_GRAY2RGB)
    if image_obj.shape[2] == 4:  # 如果是四通道 mask
        pre_mask = image_obj[:, :, 3]
        image_obj = image_obj[:, :, :3]

    Contour = get_contours(image_obj)
    Mask = np.zeros(image_obj.shape[:2], np.uint8)
    if len(Contour):
        Max_contour = Contour[0]
        Epsilon = 0.001 * cv2.arcLength(Max_contour, True)
        Approx = cv2.approxPolyDP(Max_contour, Epsilon, True)
        cv2.drawContours(Mask, [Approx], -1, 255, -1)
    else:
        Mask = np.ones(image_obj.shape[:2], np.uint8) * 255

    if pre_mask is None:
        mask = Mask
    else:
        mask = cv2.bitwise_and(Mask, pre_mask)
    return image_obj, mask


def seg_preprocess(img_path):
    print("seg_preprocess")
    img = mmcv.imread(img_path)
    ori_shape = img.shape[:2]
    img_scale = (224, 224)
    scale_factor = []
    img, x, y = mmcv.imresize(img, img_scale, return_scale=True)
    scale_factor.append(x)
    scale_factor.append(y)
    img = mmcv.imnormalize(img, mean=np.array([123.675, 116.28, 103.53]), std=np.array([58.395, 57.12, 57.375]), to_rgb=True)
    preprocessed_img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    return preprocessed_img, ori_shape


def seg_postprocess(output, ori_shape):
    seg_logit = F.interpolate(output, size=ori_shape, scale_factor=None, mode='bilinear', align_corners=False)
    seg_logit = F.softmax(seg_logit, dim=1)
    seg_pred = seg_logit.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()
    return seg_pred


def seg_infer_image(image_obj):
    print("seg_infer_image")
    image, ori_shape = seg_preprocess(image_obj)
    print("seg_infer .......")
    client = httpclient.InferenceServerClient(url=f"10.1.1.240:8000")
    print("seg_infer ......1.")

    transformed_img = image.astype(np.float32)
    print("seg_infer .......2")

    # 输入集
    inputs = [
        httpclient.InferInput("seg_input__0", transformed_img.shape, datatype="FP32")
    ]
    print("seg_infer ......3.")

    inputs[0].set_data_from_numpy(transformed_img, binary_data=True)
    # 输出集
    print("seg_infer ......4.")

    outputs = [
        httpclient.InferRequestedOutput("seg_output__0", binary_data=True),
    ]
    print("seg_infer .......5")

    results = client.infer(model_name="seg_ocrnet_hr18", inputs=inputs, outputs=outputs)
    print("seg_infer ......6.")

    # 推理
    # 取结果
    inference_output1 = torch.from_numpy(results.as_numpy("seg_output__0"))
    print("seg_infer .......7")

    seg_result = seg_postprocess(inference_output1, ori_shape)
    print("seg_infer ......8.")

    print("seg_infer 完成")
    return seg_result


def remove_background(image):
    image_obj, mask = get_mask(image)
    print(2)
    seg_result = seg_infer_image(image_obj)

    temp_front = seg_result == 1
    front_mask = (mask * (temp_front + 0).astype(np.uint8))
    temp_back = seg_result == 2
    back_mask = (mask * (temp_back + 0).astype(np.uint8))

    if len(front_mask.shape) > 2:
        front_mask = front_mask[0]
    else:
        front_mask = front_mask

    if len(back_mask.shape) > 2:
        back_mask = back_mask[0]
    else:
        back_mask = back_mask

    result_mask = front_mask + back_mask
    white_background = np.ones_like(image_obj) * 255
    result_image = np.where(result_mask[:, :, None].astype(bool), image_obj, white_background)

    return Image.fromarray(result_image)


@celery.task(time_limit=10)
def generate_image(data):
    user_id = data['user_id']
    context = data['context']
    category = data['category']
    version = data['version']
    mode = data['mode']
    samples = 4
    steps = 24
    guidance_scale = 7
    seed = random.randint(0, 2000000000)
    batch_size = 1
    model_name = f"{category}_stable_diffusion"
    triton_client.get_model_metadata(model_name=model_name, model_version=version)
    triton_client.get_model_config(model_name=model_name, model_version=version)

    image = get_image(minio_client, category)

    prompt_in = tritonclient.grpc.InferInput(name="PROMPT", shape=(batch_size,), datatype="BYTES")
    samples_in = tritonclient.grpc.InferInput("SAMPLES", (batch_size,), "INT32")
    steps_in = tritonclient.grpc.InferInput("STEPS", (batch_size,), "INT32")
    guidance_scale_in = tritonclient.grpc.InferInput("GUIDANCE_SCALE", (batch_size,), "FP32")
    seed_in = tritonclient.grpc.InferInput("SEED", (batch_size,), "INT64")
    input_images_in = tritonclient.grpc.InferInput("INPUT_IMAGES", image.shape, "FP16")
    images = tritonclient.grpc.InferRequestedOutput(name="IMAGES")
    mode_in = tritonclient.grpc.InferInput("MODE", (batch_size,), "INT32")

    prompt_in.set_data_from_numpy(np.asarray([context] * batch_size, dtype=object))
    samples_in.set_data_from_numpy(np.asarray([samples], dtype=np.int32))
    steps_in.set_data_from_numpy(np.asarray([steps], dtype=np.int32))
    guidance_scale_in.set_data_from_numpy(np.asarray([guidance_scale], dtype=np.float32))
    seed_in.set_data_from_numpy(np.asarray([seed], dtype=np.int64))
    input_images_in.set_data_from_numpy(image.astype(np.float16))
    mode_in.set_data_from_numpy(np.asarray([mode], dtype=np.int32))
    result_data = []

    def callback(result_data, result, error):
        if error:
            result_data.append(error)
        else:
            result_data.append(result)

    triton_client.async_infer(
        model_name=model_name,
        inputs=[prompt_in, samples_in, steps_in, guidance_scale_in, seed_in, input_images_in, mode_in],
        callback=partial(callback, result_data),
        outputs=[images],
    )
    time_out = 10
    while len(result_data) == 0 and time_out > 0:
        time_out = time_out - 1
        time.sleep(10)

    if len(result_data) == 1:
        if type(result_data[0]) == InferenceServerException:
            print(result_data[0])
    images = result_data[0].as_numpy("IMAGES")
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    url_list = []

    for i, image in enumerate(pil_images):
        # if category == "sketch":
        #     print(i)
        #     image = remove_background(np.asarray(image))
        image_data = io.BytesIO()
        image.save(image_data, format='PNG')
        image_data.seek(0)
        image_bytes = image_data.read()
        image_url = f"aida-users/{minio_client.put_object(f'aida-users', f'{user_id}/{category}/{generate_uuid()}_{i}.png', io.BytesIO(image_bytes), len(image_bytes), content_type='image/png').object_name}"
        url_list.append(image_url)
    return f"Image {data['user_id']} generated successfully : {url_list}"
