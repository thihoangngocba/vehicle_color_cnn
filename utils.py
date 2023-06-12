import random
import time

import albumentations as A
import cv2
import numpy as np


def random_rotate_flip(img):
    rand_rota = random.randint(-1,1)
    rand_flip = random.randint(0,1)
    if rand_rota == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rand_rota == -1:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if rand_flip > 0:
        if rand_rota != 0:
            img = cv2.flip(img, 0)
        else:
            img = cv2.flip(img, 1)
    
    return img


def horizontal_flip(img, p=0.5):
    rand_flip = random.random()
    if rand_flip < p:
        img = cv2.flip(img, 1)
    return img


def distance(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


def random_spot(image):
    img = image.copy()
    if (random.randint(0,1)==1):
        color = random.randint(0,3)
        if color==0: #'r':
            r_value = random.randint(128, 240)
            color_value = np.array([r_value, 0, 0], dtype=np.uint8)
        elif color==1: #'y':
            y_value = random.randint(128, 240)
            color_value = np.array([y_value, y_value, 0], dtype=np.uint8)        
        elif color==2: #'g':
            g_value = random.randint(128, 240)
            color_value = np.array([0, g_value, 0], dtype=np.uint8)
        elif color==3: #'w':
            w_value = random.randint(128, 240)
            color_value = np.array([w_value, w_value, w_value], dtype=np.uint8)   

        row, col, _ = img.shape
        d = min(row, col)//10
        filter = img.copy().astype(np.uint32)
        
        random_rc = list(range(0, img.shape[0]))
        random_cc = list(range(0, img.shape[1]))
        
        r_c = random.choice(random_rc)
        c_c = random.choice(random_cc)
        for i in range(row):
            for j in range(col):
                filter[i, j, :] = np.exp(-distance((i, j), (r_c, c_c))**2/(2*d**2))*color_value
        img = img+filter
        img[np.where(img>255)] = 255
        img = img.astype(np.uint8)
    return img


def add_noise(img, r_c, c_c, color=255):
    row, col, _ = img.shape
    d = min(row, col)//1
    filter = img.copy().astype(np.uint32)
    for i in range(row):
        for j in range(col):
            filter[i, j, :] = np.exp(-distance((i, j), (r_c, c_c))**2/(2*d**2))*color
    img = img+filter
    img[np.where(img>255)] = 255
    return img


def flip_image(image, raito=2, kind=0):
    h = int(image.shape[0]*raito)
    w = int(image.shape[1]*raito)
    img_flip = np.zeros((h, w, 3), dtype=np.uint8)

    if kind==0: #"tl":
        img_flip[h-image.shape[0]:, w-image.shape[1]:] = image.copy()
    elif kind==1: #"br":
        img_flip[:image.shape[0], :image.shape[1]] = image.copy()
    elif kind==2: #"tr":
        img_flip[h-image.shape[0]:, :image.shape[1]] = image.copy()
    elif kind==3: #"bl":
        img_flip[:image.shape[0], w-image.shape[1]:] = image.copy()

    return img_flip


def crop_initalize_image(image, image_flip, kind=0):
    h, w, _ = image_flip.shape
    if kind==0: #"tl":
        img = image_flip[h-image.shape[0]:, w-image.shape[1]:]
    elif kind==1: #"br":
        img = image_flip[:image.shape[0], :image.shape[1]]
    elif kind==2: #"tr":
        img = image_flip[h-image.shape[0]:, :image.shape[1]]
    elif kind==3: #"bl":
        img = image_flip[:image.shape[0], w-image.shape[1]:]

    return img


def get_random_color_value(kind):
    if kind==0: #'r':
        color = np.array([random.randint(20, 100), 0, 0], dtype=np.uint8)
    elif kind==1: #'y':
        y_value = random.randint(20, 100)
        color = np.array([y_value, y_value, 0], dtype=np.uint8)
    elif kind==2: #'g':
        color = np.array([0, random.randint(20, 100), 0], dtype=np.uint8)

    return color


def random_glow(image):
    img = image.copy()
    if (random.randint(0,1)==1):
        kind = random.randint(0,3)
        color = get_random_color_value(random.randint(0,2))
        
        img = flip_image(img, 1.2, kind=kind)

        if kind==0: #"tl":
            img = add_noise(img, 0, 0, color=color)
        elif kind==1: #"br":
            img = add_noise(img, img.shape[0], img.shape[1], color=color)
        elif kind==2: #"tr":
            img = add_noise(img, 0, img.shape[1], color=color)
        elif kind==3: #"bl":
            img = add_noise(img, img.shape[0], 0, color=color)

        img = img.astype(np.uint8)

        img = crop_initalize_image(image, img, kind=kind)
    return img


def random_ir_light(image):
    if (random.randint(0,1)==1):
        p_value = random.randint(20, 30)
        tint_mask = np.ones_like(image)*(p_value, 0, p_value)
        tint_img = image+tint_mask
        tint_img[tint_img>255]=255
        tint_img = tint_img.astype(np.uint8)
        return tint_img
    return image

# Data Generate
data_transform = A.Compose([
                            # A.VerticalFlip(p=0.5),
                            # A.HorizontalFlip(p=0.5),
                            
                            # A.RandomSnow(p=0.2, snow_point_lower=0.1, snow_point_upper=0.9, brightness_coeff=1.2),
                            # A.RandomRain(p=0.2),
                            # A.MotionBlur(p=0.3, blur_limit=5),
                            
                            # A.RandomBrightnessContrast(p=1, brightness_limit=0.5, contrast_limit=0.8),
                            A.ColorJitter(p=1, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.00),
                            
                            # A.GaussNoise(p=0.5, var_limit=0.09),
                          ])


def preprocessing_image(img):
    # img = random_rotate_flip(img)
    img = horizontal_flip(img)
    # img = random_spot(img)
    # img = random_spot(img)
    # img = random_glow(img)
    img = random_ir_light(img)

    img = data_transform(image=img)['image']

    img = np.float32(img)
    # img = img / 255
    return img


def load_weights_transfer(model_old, model_new, num_skip_layers):
    num_layers = len(model_old.layers)

    if num_skip_layers > num_layers:
        print("num_skip_layers must be less than number of model_old's layers")
        return False
    else:
        for i in range(0, num_layers-num_skip_layers):
            model_new.layers[i].set_weights(model_old.layers[i].get_weights())
        return True
    

def save_dataloader_img(save_path, raw_img):
    img = raw_img.copy()
    img = img * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    img = cv2.imread('data/y.jpeg')
    out_img = preprocessing_image(img)
    cv2.imwrite('out.jpg', out_img)
