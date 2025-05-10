from transformers import AutoModel, AutoTokenizer
from model.models import MLP_Head
from model.utils import *
import torch
from tqdm import tqdm
import numpy as np
import random
import os
import requests
import time
import argparse

def parse_arguments():
    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", help='',
                        default="./base_models/BioLinkBERT-base")
    parser.add_argument("--train_data_path", help='',
                        default='./data/train/Pubtator/train_dev_set.pubtator')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    base_model_path = args.base_model_path
    train_data_path = args.train_data_path
    tag = 'traindev'

    num_samples = 100
    log_interval = 100
    val_interval = 500

    model_name = base_model_path.strip('/').split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    pad_id = tokenizer(tokenizer.pad_token, add_special_tokens=False).data['input_ids'][0]

    all_train_data = generate_data(train_data_path, tokenizer)
    all_val_data = generate_data('./data/eval/dev_set.pubtator', tokenizer)

    f = open('test.txt', 'w', encoding='utf-8')
    for item in all_val_data:
        f.write(str(item['label']) + '\t' + tokenizer.decode(item['tokens']) + '\n')

    train_config = {
        "num_classes": 2,
        "number_of_head_layers": 2,
        "hidden_dropout_prob": 0.1,
        "weight_decay": 0.1,
        "encoder_lr": 2e-5,
        "classifier_lr": 5e-5,
        "num_epoches": 5,
        "batch_size": 32 if model_name == 'xlm-roberta-large-english-clinical' else 64,
        "use_class_weight": True
    }
    batch_size = train_config['batch_size']
    encoder = AutoModel.from_pretrained(base_model_path).bfloat16().cuda()
    classfier = MLP_Head(train_config, 1024 if model_name == 'xlm-roberta-large-english-clinical' else 768).bfloat16().cuda()

    param_dict = {pn: p for pn, p in encoder.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': train_config['weight_decay']},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    encoder_optimizer = torch.optim.AdamW(optim_groups, betas=[0.9,0.999], lr=train_config['encoder_lr'])
    head_optimizer = torch.optim.AdamW(list(classfier.parameters()), lr=train_config['classifier_lr'], weight_decay = train_config['weight_decay'])

    cnt = {0:0, 1:0}
    for data in all_train_data:
        cnt[data['label']] += 1
    class_weight = torch.tensor([cnt[key] for key in cnt], dtype=torch.bfloat16)
    class_weight = torch.mean(class_weight) / (class_weight + 10)
    class_weight = class_weight.cuda()
    print('class_weight: {:.4f}, {:.4f}'.format(class_weight[0].item(), class_weight[1].item()))

    num_steps = int(len(all_train_data) // batch_size * train_config['num_epoches'])
    print('Num steps: ', num_steps)

    cur_encoder_lr = train_config['encoder_lr']
    cur_head_lr = train_config['classifier_lr']
    lr_change_interval = num_steps // 100
    if lr_change_interval == 0:
        lr_change_interval = 1
    lr_change_gamma = 0.98

    last_t = time.time()
    for step in range(num_steps):
        batch = []
        l_batch = []
        max_len = 0
        for i in range(batch_size):
            id = random.randint(0, len(all_train_data) - 1)
            batch.append(all_train_data[id]['tokens'])
            l_batch.append(all_train_data[id]['label'])

        batch = pad(batch, pad_id)
        batch = torch.tensor(batch).cuda()
        l_batch = torch.tensor(l_batch).cuda()

        mask = torch.ones_like(batch)
        mask[batch == pad_id] = 0
        output_embed = encoder(batch, attention_mask=mask)[0][:, 0, :]
        logits, _, _ = classfier(output_embed)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), l_batch.view(-1),
                                                 reduction='none', ignore_index=-1)
        if train_config['use_class_weight']:
            all_preds = torch.argmax(logits, dim=-1)
            loss = loss * (class_weight[all_preds.view(-1)] + class_weight[l_batch.view(-1)]) / 2
        loss = loss.mean()
        encoder_optimizer.zero_grad()
        head_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        head_optimizer.step()

        if (step + 1) % log_interval == 0:
            torch.cuda.synchronize()
            dt = time.time() - last_t
            last_t = time.time()
            print(f"iter {step + 1}: loss {loss.item():.4f}, time {dt * 1000:.2f}ms")

        if (step + 1) % lr_change_interval == 0:
            cur_encoder_lr *= lr_change_gamma
            cur_head_lr *= lr_change_gamma
            for g in encoder_optimizer.param_groups:
                g['lr'] = cur_encoder_lr
            for g in head_optimizer.param_groups:
                g['lr'] = cur_head_lr

        if (step + 1) % val_interval == 0:
            tp = 0
            fp = 0
            fn = 0
            for iter in tqdm(range(num_samples)):
                batch = []
                l_batch = []
                max_len = 0
                for i in range(batch_size):
                    id = random.randint(0, len(all_val_data) - 1)
                    batch.append(all_val_data[id]['tokens'])
                    l_batch.append(all_val_data[id]['label'])

                batch = pad(batch, pad_id)
                batch = torch.tensor(batch).cuda()
                l_batch = torch.tensor(l_batch).cuda()

                mask = torch.ones_like(batch)
                mask[batch == pad_id] = 0
                output_embed = encoder(batch, attention_mask=mask)[0][:, 0, :]
                logits, _, _ = classfier(output_embed)
                preds = torch.argmax(logits, -1)
                for pred, gt in zip(preds, l_batch):
                    if pred.item() == 1 and gt.item() == 1:
                        tp += 1
                    elif pred.item() == 1:
                        fp += 1
                    elif gt.item() == 1:
                        fn += 1

            precision, recall, f_score = prf_metrics(tp,fp,fn)
            print(f"val precision: {precision:.4f}, val recall: {recall:.4f}, val f_score: {f_score:.4f}")

    save_path = './finetuned_models/finetuned_classifiers/' + model_name + '_' + tag + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    encoder.save_pretrained(save_path)
    torch.save({
        'base_model_path': base_model_path,
        'classfier_state_dict': classfier.state_dict(),
        'train_config': train_config
    }, save_path + 'saved.pt')






