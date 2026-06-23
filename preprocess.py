import numpy as np
import torch
import pandas as pd
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
import os
from tqdm import tqdm
from argparse import ArgumentParser

from faceevolve.align_trans import get_reference_facial_points, warp_and_crop_face

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)


def get_fold(subject, folds):
    for i in range(len(folds)):
        if subject in folds[i]:
            return i
    raise ValueError(f'Subject {subject} not found in folds!')


def mli_der(data_root, cache_root):
    splits = ['train', 'test']
    folds = [
        [19, 21, 34, 4, 18, 1, 13, 26, 25],
        [35, 7, 28, 22, 0, 5, 15, 31, 16],
        [36, 23, 32, 3, 8, 17, 27, 14, 10],
        [33, 29, 11, 24, 2, 12, 9, 20, 30, 6]
    ]
    data = []
    light_intensities = os.listdir(os.path.join(data_root, 'image data'))
    for li in light_intensities:
        files = os.listdir(os.path.join(data_root, 'image data', li))
        for file in files:
            subject, category, _ = file.removeprefix('subject').split('-')
            subject = int(subject)
            data.append((file, subject, li, category, get_fold(subject, folds)))
    df = pd.DataFrame(data, columns=['fn', 'subject', 'light', 'category', 'fold'])
    
    for fold in range(len(folds)):
        for split in splits:
            for label in df['category'].unique().tolist():
                os.makedirs(os.path.join(cache_root, 'MLI-DER', str(fold), split, label), exist_ok=True)
    
    img_size = 112
    scale = img_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing MLI-DER dataset...'):
        fn = row['fn']
        light = row['light']
        img = Image.open(os.path.join(data_root, 'image data', light, fn))
        img = img.convert('RGB')
        img = np.asarray(img)
        x = torch.from_numpy(img).to(device)
        boxes, probs, landmarks = mtcnn.detect(x, landmarks=True)
        if landmarks is not None:
            facial5points = landmarks[0]
            img = warp_and_crop_face(img, facial5points, reference, crop_size=(img_size, img_size), align_type='affine')
        else:
            img = cv2.resize(img, (img_size, img_size))
        img = Image.fromarray(img)
        
        label = row['category']
        img_name = row['light'] + '-' + fn
        for fold in range(len(folds)):
            split = 'test' if row['fold'] == fold else 'train'
            img.save(os.path.join(cache_root, 'MLI-DER', str(fold), split, label, img_name))


def kmu_fed(data_root, cache_root):
    labels = {
        'SU': 'surprise',
        'FE': 'fear',
        'DI': 'disgust',
        'HA': 'happy',
        'SA': 'sad',
        'AN': 'angry',
    }
    folds = [
        [1, 4, 7],
        [5, 2, 3],
        [6, 12, 10],
        [8, 9, 11]
    ]
    data = []
    files = os.listdir(data_root)
    for i in range(6):
        try:
            files.remove(f'1_AN_mr_00{i+1}.jpg')
        except:
            pass
    for file in files:
        subject, label, _, idx = file.removesuffix('.jpg').split('_')
        subject = int(subject)
        idx = int(idx)
        data.append((file, subject, labels[label], get_fold(subject, folds), idx))
    df = pd.DataFrame(data, columns=['fn', 'subject', 'label', 'fold', 'index'])
    
    splits = ['train', 'test']
    for fold in range(len(folds)):
        for split in splits:
            for label in labels.values():
                os.makedirs(os.path.join(cache_root, 'KMU-FED', str(fold), split, label), exist_ok=True)
    
    img_size = 112
    scale = img_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale
    
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing KMU-FED dataset...', leave=False):
        fn = row['fn']
        subject = row['subject']
        label = row['label']
        index = row['index']
        
        img = Image.open(os.path.join(data_root, fn))
        img = img.convert('RGB')
        img = np.asarray(img)
        x = torch.from_numpy(img).to(device)
        boxes, probs, landmarks = mtcnn.detect(x, landmarks=True)
        if landmarks is not None:
            facial5points = landmarks[0]
            img = warp_and_crop_face(img, facial5points, reference, crop_size=(img_size, img_size), align_type='affine')
        else:
            img = cv2.resize(img, (img_size, img_size))
        img = Image.fromarray(img)
        for fold in range(4):
            split = 'test' if row['fold'] == fold else 'train'
            img.save(os.path.join(cache_root, 'KMU-FED', str(fold), split, label, f'{subject}-{index}.jpg'))


if __name__ == '__main__':
    
    torch.backends.cudnn.benchmark = True
    
    parser = ArgumentParser()
    parser.add_argument('--cache-root', type=str, required=True)
    
    parser.add_argument('--mli-der-root', type=str, default=None)
        
    parser.add_argument('--kmu-fed-root', type=str, default=None)

    args = parser.parse_args()
    
    if args.mli_der_root is not None:
        mli_der(args.mli_der_root, args.cache_root)
    if args.kmu_fed_root is not None:
        kmu_fed(args.kmu_fed_root, args.cache_root)
