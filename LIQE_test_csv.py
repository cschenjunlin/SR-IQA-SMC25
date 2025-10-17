import os
import torch
import numpy as np
import pandas as pd
import clip
from utils import _preprocess2
import random
from itertools import product
from PIL import Image, ImageFile
import torch.nn.functional as F
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

##############################textual template####################################

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

##############################general setup####################################

seed = 20200626
num_patch = 15

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

session = str(1)
pseudo_weight = 0.01
# sriqa_weight = 0.1
liqe_model = f'LIQE-SR_{pseudo_weight}'
ckpt = f'./checkpoints/{liqe_model}.pt'
checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint['model_state_dict'])

joint_texts = torch.cat([clip.tokenize(f"a photo of {q} quality") for q in qualitys]).to(device)

preprocess2 = _preprocess2()

##############################general setup####################################


def do_batch(x, text):
    batch_size = x.size(0)
    num_patch = x.size(1)

    x = x.view(-1, x.size(2), x.size(3), x.size(4))

    logits_per_image, logits_per_text = model.forward(x, text)

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_text = logits_per_text.view(-1, batch_size, num_patch)

    logits_per_image = logits_per_image.mean(1)
    logits_per_text = logits_per_text.mean(2)

    logits_per_image = F.softmax(logits_per_image, dim=1)

    return logits_per_image, logits_per_text


def liqe_test_img(img_path):
    I = Image.open(img_path)
    if I.size[1] < 224 or I.size[0] < 224:
        scale_factor = max(224 / I.size[0], 224 / I.size[1])
        new_width = round(I.size[0] * scale_factor)
        new_height = round(I.size[1] * scale_factor)
        I = I.resize((new_width, new_height))
    I = preprocess2(I)
    I = I.unsqueeze(0)

    n_channels = 3
    kernel_h = 224
    kernel_w = 224
    if (I.size(2) >= 1024) | (I.size(3) >= 1024):
        step = 48
    else:
        step = 32
    I_patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                      n_channels,
                                                                                                      kernel_h,
                                                                                                      kernel_w)
    sel_step = I_patches.size(0) // num_patch
    sel = torch.zeros(num_patch)
    for i in range(num_patch):
        sel[i] = sel_step * i
    sel = sel.long()
    I_patches = I_patches[sel, ...].to(device)

    with torch.no_grad():
        logits_per_image, _ = do_batch(I_patches.unsqueeze(0), joint_texts)

    logits_per_image = logits_per_image.view(-1, len(qualitys))
    logits_quality = logits_per_image

    quality_prediction = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                         4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

    return quality_prediction


def liqe_test_csv():
    new_column = f'{liqe_model}'

    for idx, image_row in tqdm(df.iloc[:1056].iterrows(), total=df.shape[0]-1,
                               desc=f"Computing 2AFC scores of LIQE on RealSR-1K"):
        group = image_row['group'].upper()
        pair = image_row['sr_pair_names']
        dir_path = os.path.join('/home/user/research/LIQE-SR/RealSR-1K', f'{group}_images', pair)
        img1_path = os.path.join(dir_path, image_row['name1_column'])
        img2_path = os.path.join(dir_path, image_row['name2_column'])

        score1 = liqe_test_img(img1_path).item()
        score2 = liqe_test_img(img2_path).item()

        df.at[idx, new_column] = image_row['score1_column'] if score1 > score2 else image_row['score2_column']

    epoch_mean = df[new_column].mean()
    df.at[1056, new_column] = epoch_mean

    df.to_csv(score_file, index=False)


def metrics_compare():
    # Compute human scores
    metrics = df.columns[6:]
    for idx, images in df.iterrows():
        # df.at[idx, 'upper_limit'] = images[metrics].max()
        # df.at[idx, 'lower_limit'] = images[metrics].min()
        df.at[idx, 'human'] = max(df.at[idx, 'score1_column'], df.at[idx, 'score2_column'])

    # Compute average scores
    scores = df.columns[6:]
    means = df[scores].mean()
    # upper = means['upper_limit']
    # lower = means['lower_limit']
    # means = (means - lower) / (upper - lower)
    means = pd.DataFrame(means).T
    means.index = ['average']
    new_df = pd.concat([df, means])

    new_df.to_csv(score_file, index=True)


if __name__ == '__main__':
    score_file = './sr_subjective_results.csv'
    df = pd.read_csv(score_file)
    liqe_test_csv()
    # metrics_compare()
