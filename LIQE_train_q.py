import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import clip
import random
import time
from Fidelity_Loss import fidelity_loss
import scipy.stats
from utils import set_dataset_qonly, set_dataset_pl
from utils import _preprocess2, _preprocess3, convert_models_to_fp32
import torch.nn.functional as F
from itertools import product
import os
import pickle

##############################textual template####################################

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

pseudo_label = {'SeeSR': 10, 'ResShift': 9, 'SinSR': 8, 'DiffBIR': 7, 'StableSR': 6, 'USRGAN': 5,
             'DASR': 5, 'Real-ESRGAN': 4, 'LDL': 3, 'BSRGAN': 2, 'FeMaSR': 2, 'RGT': 1}

##############################general setup####################################

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

num_epoch = 80
bs = 32

train_patch = 3

quality_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys]).to(device)

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

pseudo_weight = 0.01
ckpt_save_path = f'./checkpoints/weightedLoss{pseudo_weight}'


def freeze_model():
    model.logit_scale.requires_grad = False


def do_batch(x, text):
    """ x.shape: (batch_size, num_patch, channels, height, weight) """
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


def train(model, best_result, best_epoch, srcc_dict):
    start_time = time.time()
    beta = 0.9
    running_loss = 0 if epoch == 0 else train_loss[-1]
    running_duration = 0.0
    num_steps_per_epoch = 200
    local_counter = epoch * num_steps_per_epoch + 1
    model.train()

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

        # loss for each dataset
        for dataset_idx, loader in enumerate(loaders, 0):
            try:
                sample_batched = next(loader)
            except StopIteration:
                loader = iter(train_loaders[dataset_idx])
                sample_batched = next(loader)
                loaders[dataset_idx] = loader

            x, gmos = sample_batched['I'], sample_batched['mos']
            x = x.to(device)
            gmos = gmos.to(device)

            logits_per_image, _ = do_batch(x, quality_texts)
            logits_per_image = logits_per_image.view(-1, len(qualitys))
            logits_quality = 1 * logits_per_image[:, 0] + 2 * logits_per_image[:, 1] + 3 * logits_per_image[:, 2] + \
                             4 * logits_per_image[:, 3] + 5 * logits_per_image[:, 4]

            cur_loss = fidelity_loss(logits_quality, gmos.detach()).mean()
            all_loss.append(cur_loss)

        # weighted total loss
        total_loss = 0
        loss_weights = [1.0, 1.0, pseudo_weight]
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
        srcc1 = eval(koniq10k_val_loader, phase='val', dataset='koniq10k')
        srcc11 = eval(koniq10k_test_loader, phase='test', dataset='koniq10k')

        srcc2 = eval(pipal_val_loader, phase='val', dataset='pipal')
        srcc22 = eval(pipal_test_loader, phase='test', dataset='pipal')

        srcc3 = eval_group(pl_val_loader, phase='val', dataset='pseudo-label')
        srcc33 = eval_group(pl_test_loader, phase='test', dataset='pseudo-label')

        quality_result['val'] = {'koniq10k': srcc1, 'pipal': srcc2, 'pseudo-label': srcc3}
        quality_result['test'] = {'koniq10k': srcc11, 'pipal': srcc22, 'pseudo-label': srcc33}

        all_result['val'] = {'quality': quality_result['val']}
        all_result['test'] = {'quality': quality_result['test']}

        srcc_avg = (srcc1 + srcc2 + srcc3) / 3

        if srcc_avg > best_result['quality']:
            print('**********New quality best!**********')
            best_epoch['quality'] = epoch
            best_result['quality'] = srcc_avg
            srcc_dict1['koniq10k'] = srcc11
            srcc_dict1['pipal'] = srcc22
            srcc_dict1['pseudo-label'] = srcc33

            ckpt_name = os.path.join(ckpt_save_path, str(session + 1), 'quality_best_ckpt.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'all_results': all_result
            }, ckpt_name)  # just change to your preferred folder/filename

    return best_result, best_epoch, srcc_dict, all_result


def eval(loader, phase, dataset):
    model.eval()
    q_mos = []
    q_hat = []
    for step, sample_batched in enumerate(loader, 0):

        x, gmos = sample_batched['I'], sample_batched['mos']

        x = x.to(device)
        q_mos = q_mos + gmos.cpu().tolist()

        # Calculate features
        with torch.no_grad():
            logits_per_image, _ = do_batch(x, quality_texts)

        logits_per_image = logits_per_image.view(-1, len(qualitys))

        quality_preds = 1 * logits_per_image[:, 0] + 2 * logits_per_image[:, 1] + 3 * logits_per_image[:, 2] + \
                        4 * logits_per_image[:, 3] + 5 * logits_per_image[:, 4]

        q_hat = q_hat + quality_preds.cpu().tolist()

    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)

    return srcc


def eval_group(loader, phase, dataset):
    model.eval()
    srcc = 0
    num_batches = 0
    for step, group_samples in enumerate(loader, 0):

        sample_batched = group_samples

        x, gmos = sample_batched['I'], sample_batched['mos']

        x = x.to(device)
        q_mos = gmos.cpu().tolist()

        # Calculate features
        with torch.no_grad():
            logits_per_image, _ = do_batch(x, quality_texts)

        logits_per_image = logits_per_image.view(-1, len(qualitys))

        quality_preds = 1 * logits_per_image[:, 0] + 2 * logits_per_image[:, 1] + 3 * logits_per_image[:, 2] + \
                        4 * logits_per_image[:, 3] + 5 * logits_per_image[:, 4]

        q_hat = quality_preds.cpu().tolist()

        srcc += scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
        num_batches += 1

    srcc_avg = srcc / num_batches

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)

    return srcc_avg


num_workers = 8
for session in range(0, 1):
    weighting_method = WeightMethods(
        method='dwa',
        n_tasks=1,
        alpha=1.5,
        temp=2.0,
        n_train_batch=200,
        n_epochs=num_epoch,
        main_task=0,
        device=device
    )

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    initial_lr = 5e-6
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr,
        weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    start_epoch = 0

    freeze_model()

    best_result = {'avg': 0.0, 'quality': 0.0}
    best_epoch = {'avg': 0, 'quality': 0}

    # avg
    srcc_dict = {'koniq10k': 0.0, 'pipal': 0.0, 'pseudo-label': 0.0}

    # quality
    srcc_dict1 = {'koniq10k': 0.0, 'pipal': 0.0, 'pseudo-label': 0.0}

    # splits file of datasets
    koniq10k_train_csv = os.path.join('/home/user/research/IQA/IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_train.txt')
    koniq10k_val_csv = os.path.join('/home/user/research/IQA/IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_val.txt')
    koniq10k_test_csv = os.path.join('/home/user/research/IQA/IQA_Database/koniq-10k/splits2', str(session+1), 'koniq10k_test.txt')

    pipal_train_csv = os.path.join('/home/user/research/IQA/IQA_Database/PIPAL/splits2', str(session+1), 'pipal_train.txt')
    pipal_val_csv = os.path.join('/home/user/research/IQA/IQA_Database/PIPAL/splits2', str(session+1), 'pipal_val.txt')
    pipal_test_csv = os.path.join('/home/user/research/IQA/IQA_Database/PIPAL/splits2', str(session+1), 'pipal_test.txt')

    pl_train_csv = os.path.join('/home/user/research/SR/splits721', str(session + 1), 'pseudo_train.txt')
    pl_val_csv = os.path.join('/home/user/research/SR/splits721', str(session + 1), 'pseudo_val.txt')
    pl_test_csv = os.path.join('/home/user/research/SR/splits721', str(session + 1), 'pseudo_test.txt')

    # path to datasets
    koniq10k_set = '/home/user/research/IQA/IQA_Database/koniq-10k/'
    pipal_set = '/home/user/research/IQA/IQA_Database/PIPAL/Distortion/'
    pl_set = '/home/user/research/SR/code/'

    # dataloader of datasets
    koniq10k_train_loader = set_dataset_qonly(koniq10k_train_csv, 16, koniq10k_set, num_workers, preprocess3,
                                              train_patch, False, set=0)
    koniq10k_val_loader = set_dataset_qonly(koniq10k_val_csv, 16, koniq10k_set, num_workers, preprocess2,
                                            15, True, set=1)
    koniq10k_test_loader = set_dataset_qonly(koniq10k_test_csv, 16, koniq10k_set, num_workers, preprocess2,
                                             15, True, set=2)

    pipal_train_loader = set_dataset_qonly(pipal_train_csv, 16, pipal_set, num_workers, preprocess3,
                                           train_patch, False, set=0)
    pipal_val_loader = set_dataset_qonly(pipal_val_csv, 16, pipal_set, num_workers, preprocess2,
                                         15, True, set=1)
    pipal_test_loader = set_dataset_qonly(pipal_test_csv, 16, pipal_set, num_workers, preprocess2,
                                          15, True, set=2)

    pl_train_loader = set_dataset_pl(pl_train_csv, pl_set, num_workers, preprocess3,
                                     train_patch, False, set=0, pseudo_label=pseudo_label)
    pl_val_loader = set_dataset_pl(pl_val_csv, pl_set, num_workers, preprocess2,
                                   15, True, set=1, pseudo_label=pseudo_label)
    pl_test_loader = set_dataset_pl(pl_test_csv, pl_set, num_workers, preprocess2,
                                    15, True, set=2, pseudo_label=pseudo_label)

    train_loaders = [koniq10k_train_loader, pipal_train_loader, pl_train_loader]

    result_pkl = {}
    for epoch in range(0, num_epoch):
        best_result, best_epoch, srcc_dict, all_result = train(model, best_result, best_epoch, srcc_dict)
        scheduler.step()

        result_pkl[str(epoch)] = all_result

        print(weighting_method.method.lambda_weight[:, epoch])

        print('...............current quality best...............')
        print('best quality epoch:{}'.format(best_epoch['quality']))
        print('best quality result:{}'.format(best_result['quality']))
        for dataset in srcc_dict1.keys():
            print_text = dataset + ':' + 'srcc:{}'.format(srcc_dict1[dataset])
            print(print_text)

    pkl_name = os.path.join(ckpt_save_path, str(session+1), 'all_results.pkl')
    with open(pkl_name, 'wb') as f:
        pickle.dump(result_pkl, f)

    lambdas = weighting_method.method.lambda_weight
    pkl_name = os.path.join(ckpt_save_path, str(session+1), 'lambdas.pkl')
    with open(pkl_name, 'wb') as f:
        pickle.dump(lambdas, f)
