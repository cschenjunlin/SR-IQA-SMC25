import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
import clip
import random
import time
from Fidelity_Loss import fidelity_loss, fidelity_loss_majority_voting
import scipy.stats
from utils import set_dataset_NR, set_dataset_FR, set_dataset_csv, set_dataset_group
from utils import _preprocess2, _preprocess3, convert_models_to_fp32
from PIL import Image, ImageFile
import torch.nn.functional as F
from itertools import product
import os
import pickle
from tqdm import tqdm
from maniqa_arch import MANIQA
from rectifier import Rectifier

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

############################## textual template ####################################

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

# pseudo_label = {'SeeSR': 10, 'ResShift': 9, 'SinSR': 8, 'DiffBIR': 7, 'StableSR': 6, 'USRGAN': 5,
#                 'DASR': 5, 'Real-ESRGAN': 4, 'LDL': 3, 'BSRGAN': 2, 'FeMaSR': 2, 'RGT': 1}

############################## general setup ####################################

seed = 20200626
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# batch_size * patch_num < 90, otherwise CUDA out of memory!
train_bs = 16
test_bs = 4
train_patch = 3
test_patch = 15
num_workers = 8

initial_lr = 5e-6
num_epoch = 50

quality_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys]).to(device)

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

pseudo_label_weight = 0.01
loss_weights = [1.0, 1.0, pseudo_label_weight]
ckpt_save_path = f'./checkpoints/{pseudo_label_weight}'


def freeze_model():
    model.logit_scale.requires_grad = False


def do_batch(x, text):
    """
    Args:
        x.shape = (batch_size, num_patch, channels, height, weight).
        text.shape = (5, 77).
    Returns:
        logits_per_image.shape = (batch_size, 5)
        logits_per_text.shape = (5, batch_size)
    """
    batch_size = x.size(0)
    num_patch = x.size(1)

    x = (x - clip_mean.to(x)) / clip_std.to(x)
    x = x.view(-1, x.size(2), x.size(3), x.size(4))

    logits_per_image, logits_per_text = model.forward(x, text)

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_text = logits_per_text.view(-1, batch_size, num_patch)

    logits_per_image = logits_per_image.mean(1)
    logits_per_text = logits_per_text.mean(2)

    logits_per_image = F.softmax(logits_per_image, dim=1)

    return logits_per_image, logits_per_text


def train(model, rectifier, best_result, best_epoch, srcc_dict):
    start_time = time.time()
    beta = 0.9
    running_loss = 0 if epoch == 0 else train_loss[-1]
    running_duration = 0.0
    num_steps_per_epoch = 200
    local_counter = epoch * num_steps_per_epoch + 1

    model.train()
    rectifier.train()

    loaders = []
    for loader in train_loaders:
        loaders.append(iter(loader))

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
    for step in range(num_steps_per_epoch):
        all_loss = []

        optimizer.zero_grad()

        for dataset_idx, loader in enumerate(loaders, 0):
            try:
                sample_batched = next(loader)
            except StopIteration:
                loader = iter(train_loaders[dataset_idx])
                sample_batched = next(loader)
                loaders[dataset_idx] = loader

            # x.shape: (batch_size, num_patch, channels, height, weight)
            x, gmos = sample_batched['I'], sample_batched['mos']
            x = x.to(device)
            gmos = gmos.to(device)

            logits_per_image, _ = do_batch(x, quality_texts)

            logits_per_image = logits_per_image.view(-1, len(qualitys))
            logits_quality = (1 * logits_per_image[:, 0] + 2 * logits_per_image[:, 1] + 3 * logits_per_image[:, 2] +
                              4 * logits_per_image[:, 3] + 5 * logits_per_image[:, 4])

            # Rectify the predicted score with maniqa features of SR images.
            x_first_patch = x[:, 0, :, :, :]
            x_feat = maniqa_model(x_first_patch, train_patch)

            # a(x+1)+b, where +1 is for residual learning
            scale, shift = rectifier(x_feat)
            rectified_quality = (torch.abs(scale) + 1) * logits_quality + shift

            if dataset_idx < 2:
                cur_loss = fidelity_loss(rectified_quality, gmos.detach()).mean()
            else:
                cur_loss = fidelity_loss_majority_voting(rectified_quality, gmos.detach()).mean()

            all_loss.append(cur_loss)

        # weighted total loss
        total_loss = 0
        for idx, (loss, weight) in enumerate(zip(all_loss, loss_weights)):
            total_loss += loss * weight

        total_loss.backward()

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        # statistics
        running_loss = beta * running_loss + (1 - beta) * total_loss.data.item()
        loss_corrected = running_loss / (1 - beta ** local_counter)

        current_time = time.time()
        duration = current_time - start_time
        running_duration = beta * running_duration + (1 - beta) * duration
        duration_corrected = running_duration / (1 - beta ** local_counter)
        examples_per_sec = x.size(0) / duration_corrected
        format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (epoch, step + 1, num_steps_per_epoch, loss_corrected,
                            examples_per_sec, duration_corrected))

        local_counter += 1
        start_time = time.time()

        train_loss.append(loss_corrected)

    quality_result = {'val': {}, 'test': {}}
    all_result = {'val': {}, 'test': {}}

    if (epoch >= 0):
        # srcc1 = eval(koniq10k_val_loader, phase='val', dataset='koniq10k')
        srcc11 = eval(koniq10k_test_loader, phase='test', dataset='koniq10k')

        # srcc2 = eval(pipal_val_loader, phase='val', dataset='pipal')
        srcc22 = eval(pipal_val_loader, phase='test', dataset='pipal')

        # srcc3 = eval_group(pseudo_label_val_loader, phase='val', dataset='pseudo-label')
        # srcc33 = eval_group(pseudo_label_test_loader, phase='test', dataset='pseudo-label')

        srcc44 = eval(qads_loader, phase='test', dataset='qads')

        srcc55 = eval(ma17_loader, phase='test', dataset='ma17')

        srcc66 = eval_group(sriqa_bench_loader, phase='test', dataset='sriqa_bench')

        # quality_result['val'] = {'koniq10k': srcc1}
        quality_result['test'] = {'koniq10k': srcc11, 'pipal': srcc22,
                                  'qads': srcc44, 'ma17': srcc55, 'sriqa_bench': srcc66}

        all_result['val'] = {'quality': quality_result['val']}
        all_result['test'] = {'quality': quality_result['test']}

        # srcc_avg = (srcc1 + srcc2) / 2

        save_ckpt = liqe_test_csv()

        if save_ckpt:
        # if srcc_avg > best_result['quality']:
            print('**********New quality best!**********')
            best_epoch['quality'] = epoch
            # best_result['quality'] = srcc_avg
            srcc_dict1['koniq10k'] = srcc11
            srcc_dict1['pipal'] = srcc22
            # srcc_dict1['pseudo-label'] = srcc33
            srcc_dict1['qads'] = srcc44
            srcc_dict1['ma17'] = srcc55
            srcc_dict1['sriqa_bench'] = srcc66

            ckpt_name = os.path.join(ckpt_save_path, str(session + 1), f'LIQE-SR_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'rectifier_state_dict': rectifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'all_results': all_result
            }, ckpt_name)  # just change to your preferred folder/filename

    return best_result, best_epoch, srcc_dict, all_result


def eval(loader, phase, dataset):
    model.eval()
    rectifier.eval()

    q_mos = []
    q_hat = []
    for step, sample_batched in enumerate(loader, 0):

        # x.shape = (batch_size, num_patch, channels, height, weight)
        x, gmos = sample_batched['I'], sample_batched['mos']
        x = x.to(device)
        q_mos = q_mos + gmos.cpu().tolist()

        # Calculate features
        with torch.no_grad():
            logits_per_image, _ = do_batch(x, quality_texts)

        logits_per_image = logits_per_image.view(-1, len(qualitys))

        quality_preds = 1 * logits_per_image[:, 0] + 2 * logits_per_image[:, 1] + 3 * logits_per_image[:, 2] + \
                        4 * logits_per_image[:, 3] + 5 * logits_per_image[:, 4]

        # Rectify the predicted score with maniqa features of SR images.
        with torch.no_grad():
            batch_size = x.size(0)
            num_patch = x.size(1)

            x = x.view(-1, x.size(2), x.size(3), x.size(4))
            x_feat = maniqa_model(x, test_patch)
            x_feat = x_feat.view(batch_size, num_patch, x_feat.size(1), x_feat.size(2))
            x_feat = x_feat.mean(1)

            # a(x+1)+b, where +1 is for residual learning
            scale, shift = rectifier(x_feat)
            rectified_quality = (torch.abs(scale) + 1) * quality_preds + shift

        q_hat = q_hat + rectified_quality.cpu().tolist()

    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)

    return srcc


def eval_group(loader, phase, dataset):
    model.eval()
    rectifier.eval()

    srcc = 0
    num_batches = 0
    for step, sample_batched in enumerate(loader, 0):

        # x.shape = (batch_size, num_patch, channels, height, weight)
        x, gmos = sample_batched['I'], sample_batched['mos']
        x = x.to(device)
        q_mos = gmos.cpu().tolist()

        # Calculate features
        with torch.no_grad():
            logits_per_image, _ = do_batch(x, quality_texts)

        logits_per_image = logits_per_image.view(-1, len(qualitys))

        quality_preds = 1 * logits_per_image[:, 0] + 2 * logits_per_image[:, 1] + 3 * logits_per_image[:, 2] + \
                        4 * logits_per_image[:, 3] + 5 * logits_per_image[:, 4]

        # Rectify the predicted score with maniqa features of SR images.
        with torch.no_grad():
            batch_size = x.size(0)
            num_patch = x.size(1)

            x = x.view(-1, x.size(2), x.size(3), x.size(4))
            x_feat = maniqa_model(x, test_patch)
            x_feat = x_feat.view(batch_size, num_patch, x_feat.size(1), x_feat.size(2))
            x_feat = x_feat.mean(1)

            # a(x+1)+b, where +1 is for residual learning
            scale, shift = rectifier(x_feat)
            rectified_quality = (torch.abs(scale) + 1) * quality_preds + shift

        q_hat = rectified_quality.cpu().tolist()

        srcc += scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        num_batches += 1

    srcc_avg = srcc / num_batches

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)

    return srcc_avg


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
    sel_step = I_patches.size(0) // test_patch
    sel = torch.zeros(test_patch)
    for i in range(test_patch):
        sel[i] = sel_step * i
    sel = sel.long()
    I_patches = I_patches[sel, ...].to(device)
    x = I_patches.to(device)

    with torch.no_grad():
        logits_per_image, _ = do_batch(I_patches.unsqueeze(0), quality_texts)

    logits_per_image = logits_per_image.view(-1, len(qualitys))

    quality_prediction = 1 * logits_per_image[:, 0] + 2 * logits_per_image[:, 1] + 3 * logits_per_image[:, 2] + \
                         4 * logits_per_image[:, 3] + 5 * logits_per_image[:, 4]

    # Rectify the predicted score with maniqa features of SR images.
    with torch.no_grad():
        x_feat = maniqa_model(x, test_patch)
        x_feat = x_feat.mean(0)
        x_feat = x_feat.unsqueeze(0)

        # a(x+1)+b, where +1 is for residual learning
        scale, shift = rectifier(x_feat)
        rectified_quality = (torch.abs(scale) + 1) * quality_prediction + shift

    return rectified_quality


def liqe_test_csv():
    score_file = './sr_subjective_results.csv'
    df = pd.read_csv(score_file)
    cur_mean_max = df.iloc[-1, 7:].max()
    new_column = f'epoch{epoch+1}'

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

    if epoch_mean > cur_mean_max:
        return True
    else:
        return False


for session in range(0, 1):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    maniqa_model = MANIQA().to(device)
    maniqa_model.eval()
    rectifier = Rectifier().to(device)

    for param in maniqa_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {'params': model.parameters()},
            {'params': rectifier.parameters()}
        ],
        lr=initial_lr, weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    start_epoch = 0

    freeze_model()

    best_result = {'avg': 0.0, 'quality': 0.0}
    best_epoch = {'avg': 0, 'quality': 0}

    ############################## datasets ####################################

    # train_sets = ['koniq10k', 'pipal', 'pseudo_label']
    # val_sets = ['koniq10k', 'pipal', 'qads', 'ma17', 'sriqa_bench']
    # test_sets = ['koniq10k', 'pipal', 'qads', 'ma17', 'sriqa_bench']

    ############################## datasets ####################################

    # avg
    srcc_dict = {'koniq10k': 0.0, 'pipal': 0.0, 'pseudo-label': 0.0, 'qads': 0.0, 'ma17': 0.0, 'sriqa_bench': 0.0}

    # quality
    srcc_dict1 = {'koniq10k': 0.0, 'pipal': 0.0, 'pseudo-label': 0.0, 'qads': 0.0, 'ma17': 0.0, 'sriqa_bench': 0.0}

    # mos file of datasets
    koniq10k_train_csv = os.path.join('/home/user/research/IQA/IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_train.txt')
    koniq10k_val_csv = os.path.join('/home/user/research/IQA/IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_val.txt')
    koniq10k_test_csv = os.path.join('/home/user/research/IQA/IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_test.txt')

    pipal_train_csv = os.path.join('/home/user/research/IQA/IQA_Database/PIPAL/train_label_all.txt')
    pipal_val_csv = os.path.join('/home/user/research/IQA/IQA_Database/PIPAL/val_label.txt')
    # pipal_test_csv = os.path.join('/home/user/research/IQA/IQA_Database/PIPAL/splits2', str(session+1), 'pipal_test.txt')

    pseudo_label_train_csv = '/home/user/research/0421/nr_test_sr_all.csv'
    # pseudo_label_val_csv = os.path.join('/home/user/research/SR/splits721', str(session + 1), 'pseudo_val.txt')
    # pseudo_label_test_csv = os.path.join('/home/user/research/SR/splits721', str(session + 1), 'pseudo_test.txt')

    qads_csv = os.path.join('/home/user/research/IQA/IQA_Database/QADS/QADS_mos.txt')

    ma17_csv = os.path.join('/home/user/research/IQA/IQA_Database/SRimages/sr_metric_data/Ma_mos.txt')

    sriqa_bench_csv = os.path.join('/home/user/research/IQA/IQA_Database/SRIQA-Bench/MOS')

    # path to datasets
    koniq10k_set = '/home/user/research/IQA/IQA_Database/koniq-10k/'

    pipal_train_set = '/home/user/research/IQA/IQA_Database/PIPAL/Train_Dist/'
    pipal_train_ref_set = '/home/user/research/IQA/IQA_Database/PIPAL/Train_Ref/'
    pipal_val_set = '/home/user/research/IQA/IQA_Database/PIPAL/Val_Dist/'
    pipal_val_ref_set = '/home/user/research/IQA/IQA_Database/PIPAL/Val_Ref/'

    pseudo_label_set = '/home/user/research/SR/code/'
    # pseudo_label_lr_set = '/home/user/research/SR/LR/'

    qads_set = '/home/user/research/IQA/IQA_Database/QADS/super-resolved_images'

    ma17_set = '/home/user/research/IQA/IQA_Database/SRimages'

    sriqa_bench_set = '/home/user/research/IQA/IQA_Database/SRIQA-Bench/TestImages'

    # dataloader of datasets
    koniq10k_train_loader = set_dataset_NR(koniq10k_train_csv, train_bs, koniq10k_set, num_workers,
                                           preprocess3, train_patch, False, set=0)
    koniq10k_val_loader = set_dataset_NR(koniq10k_val_csv, test_bs, koniq10k_set, num_workers,
                                         preprocess2, test_patch, True, set=1)
    koniq10k_test_loader = set_dataset_NR(koniq10k_test_csv, test_bs, koniq10k_set, num_workers,
                                          preprocess2, test_patch, True, set=2)

    pipal_train_loader = set_dataset_FR(pipal_train_csv, train_bs, pipal_train_set, pipal_train_ref_set, num_workers,
                                        preprocess3, train_patch, False, set=0)
    pipal_val_loader = set_dataset_FR(pipal_val_csv, test_bs, pipal_val_set, pipal_val_ref_set, num_workers,
                                      preprocess2, test_patch, True, set=1)
    # pipal_test_loader = set_dataset_FR(pipal_test_csv, test_bs, pipal_set, pipal_ref_set, num_workers,
    #                                    preprocess2, test_patch, True, set=2)

    pseudo_label_train_loader = set_dataset_csv(pseudo_label_train_csv, train_bs, pseudo_label_set, num_workers,
                                                preprocess3, train_patch, False)
    # pseudo_label_val_loader = set_dataset_pseudo_label(pseudo_label_val_csv, pseudo_label_set, num_workers,
    #                                                    preprocess2, test_patch, True, set=1)
    # pseudo_label_test_loader = set_dataset_pseudo_label(pseudo_label_test_csv, pseudo_label_set, num_workers,
    #                                                     preprocess2, test_patch, True, set=2)

    qads_loader = set_dataset_NR(qads_csv, test_bs, qads_set, num_workers, preprocess2, test_patch, True, set=2)

    ma17_loader = set_dataset_NR(ma17_csv, test_bs, ma17_set, num_workers, preprocess2, test_patch, True, set=2)

    sriqa_bench_loader = set_dataset_group(sriqa_bench_csv, sriqa_bench_set, num_workers, preprocess2, 8, True)

    train_loaders = [koniq10k_train_loader, pipal_train_loader, pseudo_label_train_loader]

    result_pkl = {}
    for epoch in range(0, num_epoch):
        best_result, best_epoch, srcc_dict, all_result = train(model, rectifier, best_result, best_epoch, srcc_dict)
        scheduler.step()

        result_pkl[str(epoch)] = all_result

        # print(weighting_method.method.lambda_weight[:, epoch])

        print('...............current quality best...............')
        print('best quality epoch:{}'.format(best_epoch['quality']))
        # print('best quality result:{}'.format(best_result['quality']))
        for dataset in srcc_dict1.keys():
            print_text = dataset + ':' + 'srcc:{}'.format(srcc_dict1[dataset])
            print(print_text)

    pkl_name = os.path.join(ckpt_save_path, str(session+1), 'all_results.pkl')
    with open(pkl_name, 'wb') as f:
        pickle.dump(result_pkl, f)

    # lambdas = weighting_method.method.lambda_weight
    # pkl_name = os.path.join(ckpt_save_path, str(session+1), 'lambdas.pkl')
    # with open(pkl_name, 'wb') as f:
    #     pickle.dump(lambdas, f)
