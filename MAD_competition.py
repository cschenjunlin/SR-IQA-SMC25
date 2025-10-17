import os
import csv
import pandas as pd
import shutil
import torch
import torch.nn.functional as F
from torchvision.models import get_model, VGG16_BN_Weights
from pyiqa import create_metric
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.img_util import imread2tensor
from tqdm import tqdm

lr = "./LR"
methods_path = "./code"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


def extract_suffix(sr_dir):
    full_name = os.listdir(sr_dir)[0]
    method_name = full_name.split('_')[-1].replace('.png', '')
    suffix = '_' + method_name
    return suffix


def MAD1(metric_name):
    """
    For each pair of methods, select LR images that maximizes D1 of the resulting super-resolution images.
    """
    assert DEFAULT_CONFIGS[metric_name]['metric_mode'] == 'FR', f"{metric_name} is not an FR method."

    D1 = create_metric(metric_name).to(device)
    print("lower_better:", D1.lower_better)

    methods = os.listdir(methods_path)
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1 = methods[i]
            sr1 = os.path.join(methods_path, method1, "SR")
            suffix1 = extract_suffix(sr1)
            img_dir1 = sorted(os.listdir(sr1),
                              key=lambda x: x.replace(suffix1, ''))

            method2 = methods[j]
            sr2 = os.path.join(methods_path, method2, "SR")
            suffix2 = extract_suffix(sr2)
            img_dir2 = sorted(os.listdir(sr2),
                              key=lambda x: x.replace(suffix2, ''))

            cur_pair = method1 + '_' + method2
            D1_file = os.path.join("MAD/D1_files/", cur_pair + ".csv")
            MAD1_file = os.path.join("MAD/MAD_files/MAD1_files/", cur_pair + ".csv")

            if os.path.exists(D1_file):
                df = pd.read_csv(D1_file)
                if len(df) == len(img_dir1):
                    continue

            with open(D1_file, 'w') as f:
                csvwriter = csv.writer(f)
                header = ['img_name', 'd1']
                csvwriter.writerow(header)

                for k in tqdm(range(len(img_dir1)), desc=f"Calculating d1 of {cur_pair}"):
                    img1_path = os.path.join(sr1, img_dir1[k])
                    img2_path = os.path.join(sr2, img_dir2[k])
                    img_name = img_dir1[k].replace(suffix1, '')

                    d1 = D1(img1_path, img2_path).item()
                    csvwriter.writerow([img_name, d1])

            df = pd.read_csv(D1_file)
            df_MAD = df.sort_values(by='d1', ascending=False)
            df_MAD.to_csv(MAD1_file, index=False)


def MAD(lamb=0.5):
    """
    For each pair of methods, select LR images that maximizes D of the resulting super-resolution images,
    where D = D1 + lamb * D2.
    """
    D2 = get_model('vgg16_bn', weights=VGG16_BN_Weights.DEFAULT, num_classes=1000).to(device)
    D2.eval()
    preprocess = VGG16_BN_Weights.DEFAULT.transforms()

    # VGG16 extracting LR image features
    feature = {}
    for filename in tqdm(os.listdir(lr), desc=f"Extracting VGG16 features of {lr}"):
        img_path = os.path.join(lr, filename)
        with torch.no_grad():
            img = preprocess(imread2tensor(img_path, rgb=True)).unsqueeze(0)
            feat = D2.features(img.to(device))
        img_name = filename.replace('.png', '')
        feature[img_name] = feat

    methods = os.listdir(methods_path)
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            cur_pair = method1 + '_' + method2
            MAD1_file = os.path.join("MAD/MAD_files/MAD1_files/", cur_pair + ".csv")
            MAD_file = os.path.join(f"MAD/MAD_files/lambda={lamb}", cur_pair + ".csv")

            if os.path.exists(MAD_file):
                df = pd.read_csv(MAD_file)
                if len(df) == 12:
                    continue

            # Reading d1 file and initializing d2
            d1 = {}
            d2 = {}
            df = pd.read_csv(MAD1_file)
            for _, img in df.iterrows():
                d1[img['img_name']] = img['d1']
                d2[img['img_name']] = 0

            with open(MAD_file, 'w') as f:
                csvwriter = csv.writer(f)
                header = ['img_name', 'd1', 'd2']
                csvwriter.writerow(header)

                # Selecting the first image with max d1
                xk = df.loc[0, 'img_name']
                csvwriter.writerow([xk, d1[xk], ''])
                S = [xk, ]
                del d2[xk]

                # Selecting top-12 MAD images
                for k in tqdm(range(1, 12), desc=f"Selecting MAD images of {cur_pair}"):
                    for x in d2.keys():
                        d2[x] += F.mse_loss(feature[x], feature[xk]).item()
                    xk = max(d2.keys(), key=lambda x: d1[x] + lamb * d2[x] / k)
                    csvwriter.writerow([xk, d1[xk], lamb * d2[xk] / k])
                    S.append(xk)
                    del d2[xk]


def select_images(file_path, image_path):
    methods = os.listdir(methods_path)
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            sr1 = os.path.join(methods_path, method1, "SR")
            sr2 = os.path.join(methods_path, method2, "SR")

            cur_pair = method1 + '_' + method2
            os.makedirs(os.path.join(image_path, cur_pair), exist_ok=True)
            MAD_file = os.path.join(file_path, cur_pair + ".csv")

            df = pd.read_csv(MAD_file, dtype=str)
            for k in tqdm(range(12), desc=f"Selecting images of {cur_pair}"):
                img_name = df.loc[k, 'img_name']
                img1 = next(file for file in os.listdir(sr1)
                            if file.startswith((f"{img_name}.", f"{img_name}_")))
                img2 = next(file for file in os.listdir(sr2)
                            if file.startswith((f"{img_name}.", f"{img_name}_")))
                shutil.copy2(os.path.join("bicubic", img_name + '.png'),
                             os.path.join(image_path, cur_pair, img_name + '_' + 'bicubic' + '.png'))
                shutil.copy2(os.path.join(sr1, img1),
                             os.path.join(image_path, cur_pair, img_name + '_' + method1 + '.png'))
                shutil.copy2(os.path.join(sr2, img2),
                             os.path.join(image_path, cur_pair, img_name + '_' + method2 + '.png'))


if __name__ == "__main__":
    MAD1('dists')
    MAD(0.05)
    select_images("MAD/MAD_files/lambda=0.05", "MAD/lambda=0.05")
