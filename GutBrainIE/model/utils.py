import torch
import numpy as np
import faiss
from tqdm import tqdm
import pandas as pd
import os
POS_CONST = 1000000
def generate_data(file, tokenizer):
    lines = open('./data/relation_set.txt', 'r', encoding='utf-8').readlines()
    all_valid_relations = []
    for i in range(len(lines)):
        items = lines[i].strip('\n').split('\t')
        if len(items) == 3:
            new_items = []
            for item in items:
                new_items.append(item.strip())
            all_valid_relations.append(new_items)
    assert len(all_valid_relations) == 54

    sep_token = tokenizer.sep_token
    bos_token = tokenizer.bos_token if hasattr(tokenizer,
                                               'bos_token') and tokenizer.bos_token is not None else tokenizer.cls_token
    eos_token = tokenizer.eos_token if hasattr(tokenizer,
                                               'eos_token') and tokenizer.eos_token is not None else tokenizer.sep_token
    sep_token = tokenizer(sep_token, add_special_tokens=False).data['input_ids'][0]
    bos_token = tokenizer(bos_token, add_special_tokens=False).data['input_ids'][0]
    eos_token = tokenizer(eos_token, add_special_tokens=False).data['input_ids'][0]

    f = open(file, 'r', encoding='utf-8')
    content = f.readlines()
    f.close()

    all_data = []
    all_ids = []
    all_texts = []
    all_annos = []
    all_res = []
    pos = 0
    while pos < len(content):
        if '|t|' in content[pos]:
            all_ids.append(content[pos].split('|t|')[0])
            text = content[pos].split('|t|')[-1] + content[pos+1].split('|a|')[-1].strip('\n')
            pos += 2

            annotations = []
            relations = {}
            while pos < len(content) and content[pos] != '\n':
                items = content[pos].split('\t')
                if len(items) == 5:
                    items.append('None')
                if items[1] == 'relation':
                    relations[(int(items[3]),int(items[4]))] = items[2]
                else:
                    annotations.append([int(items[1]), int(items[2]), items[3], items[4].strip('\n')])
                pos += 1

            all_texts.append(text)
            all_annos.append(annotations)
            all_res.append(relations)
        else:
            pos += 1

    for id, text, annos, relations in tqdm(zip(all_ids, all_texts, all_annos, all_res), total=len(all_texts)):
        sentences = text.split('.')
        for i in range(len(sentences)):
            sentences[i] += '.'

        for re in all_valid_relations:
            for i in range(len(annos)):
                if annos[i][3] == re[0]:
                    for j in range(len(annos)):
                        if j!=i and annos[j][3] == re[1]:
                            first = i
                            second = j
                            if annos[i][0] > annos[j][0]:
                                first = j
                                second = i
                            if annos[second][0] - annos[first][0] > 200:
                                continue
                            pos1 = -1
                            pos2 = -1
                            tot_len = 0
                            tot_lens = [0]
                            for k in range(len(sentences)):
                                tot_len += len(sentences[k])
                                tot_lens.append(tot_len)
                                if pos1 == -1 and tot_len >= annos[first][0] + 1:
                                    pos1 = k
                                if pos2 == -1 and tot_len >= annos[second][0] + 1:
                                    pos2 = k
                            assert pos1 >= 0 and pos2 >= 0

                            ent_tokens1 = tokenizer(annos[first][2], add_special_tokens=False).data['input_ids']
                            ent_tokens2 = tokenizer(annos[second][2], add_special_tokens=False).data['input_ids']
                            re_tokens1 = tokenizer(annos[first][2] + ' ' + re[2] + ' ' + annos[second][2],add_special_tokens=False).data['input_ids']
                            re_tokens2 = tokenizer(annos[second][2] + ' ' + re[2] + ' ' + annos[first][2],add_special_tokens=False).data['input_ids']
                            if pos1 == pos2:
                                inv1 = sentences[pos1][:annos[first][0] - tot_lens[pos1]]
                                inv2 = sentences[pos1][annos[first][1] + 1 - tot_lens[pos1]:annos[second][0] - tot_lens[pos1]]
                                inv3 = sentences[pos1][annos[second][1] + 1 - tot_lens[pos1]:]
                                tokens1 = [] if inv1 == '' else tokenizer(inv1, add_special_tokens=False).data['input_ids']
                                tokens2 = [] if inv2 == '' else tokenizer(inv2, add_special_tokens=False).data['input_ids']
                                tokens3 = [] if inv3 == '' else tokenizer(inv3, add_special_tokens=False).data['input_ids']
                                text_tokens = tokens1 + [sep_token] + ent_tokens1 + [sep_token] + tokens2 + [sep_token] + ent_tokens2 + [sep_token] + tokens3
                            else:
                                inv1 = sentences[pos1][:annos[first][0] - tot_lens[pos1]]
                                inv2 = sentences[pos1][annos[first][1] + 1 - tot_lens[pos1]:]
                                tokens1 = [] if inv1 == '' else tokenizer(inv1, add_special_tokens=False).data['input_ids']
                                tokens2 = [] if inv2 == '' else tokenizer(inv2, add_special_tokens=False).data['input_ids']
                                start_tokens = tokens1 + [sep_token] + ent_tokens1 + [sep_token] + tokens2

                                inv1 = sentences[pos2][:annos[second][0] - tot_lens[pos2]]
                                inv2 = sentences[pos2][annos[second][1] + 1 - tot_lens[pos2]:]
                                tokens1 = [] if inv1 == '' else tokenizer(inv1, add_special_tokens=False).data['input_ids']
                                tokens2 = [] if inv2 == '' else tokenizer(inv2, add_special_tokens=False).data['input_ids']
                                end_tokens = tokens1 + [sep_token] + ent_tokens2 + [sep_token] + tokens2

                                text_tokens = start_tokens
                                tot_tokens = len(re_tokens1) + len(start_tokens) + len(end_tokens) + 3
                                for k in range(pos1+1, pos2):
                                    mid_tokens = tokenizer(sentences[k], add_special_tokens=False).data['input_ids']
                                    if tot_tokens + len(mid_tokens) <= 512:
                                        text_tokens += mid_tokens
                                        tot_tokens += len(mid_tokens)
                                text_tokens += end_tokens

                            data = {}
                            data['relation'] = (id, first, second, re[2])
                            data['tokens'] = [bos_token] + re_tokens1 + [sep_token] + text_tokens + [eos_token]
                            data['label'] = 1 if ((i, j) in relations and relations[(i, j)] == re[2] and first == i) else 0
                            all_data.append(data)

                            data = {}
                            data['relation'] = (id, second, first, re[2])
                            data['tokens'] = [bos_token] + re_tokens2 + [sep_token] + text_tokens + [eos_token]
                            data['label'] = 1 if ((i, j) in relations and relations[(i, j)] == re[2] and first == j) else 0
                            all_data.append(data)

    return all_data
def prf_metrics(total_tp, total_fp, total_fn):
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f_score = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f_score

def pad(batch, pad_id):
    max_len = 0
    for item in batch:
        if len(item) > max_len:
            max_len = len(item)
    for i in range(len(batch)):
        batch[i] += [pad_id for k in range(max_len - len(batch[i]))]
    return batch

def save_predictions(raw_texts, predictions, save_path, with_score = False):
    # Annotation format: [start, end, score, linking]
    file = open(save_path, "w", encoding='utf-8')
    for i in range(len(raw_texts)):
        for line in raw_texts[i]:
            file.write(line.strip('\n') + '\n')
        doc_id = raw_texts[i][0].split('|')[0]
        text = raw_texts[i][0].split('|t|')[-1].strip('\n') + ' ' + raw_texts[i][1].split('|a|')[-1].strip('\n')
        for type in predictions:
            for anno in predictions[type][i]:
                linking = anno[3] if len(anno) >= 4 else 'None'
                out_str = doc_id + '\t' + str(anno[0]) + '\t' + str(anno[1]-1) + '\t' + text[anno[0]:anno[1]] + '\t' + type + '\t' + linking
                file.write(out_str + '\t' + str(anno[2]) + '\n' if with_score else out_str + '\n')
        file.write('\n')

# Post processing to remove leading or trailing special characters in tokens and remove weak predictions
# prediction format: [[start, end, score, (optional)linking]]
def post_process(raw_text, predictions, dataset, score_threshold = 0):
    text = ''
    for item in raw_text:
        text += item.split('|a|')[1] if '|a|' in item else item.split('|t|')[1]

    strip_list = [' ', '\n', ',', '.', '-']
    opposite = {'(': ')', ')': '(', '[': ']', ']': '[', '{': '}', '}': '{'}

    # remove special characters
    new_predictions = []
    for item in predictions:
        if item[1] > len(text):
            item[1] = len(text)
        if item[0] >= item[1]:
            item[0] = item[1] - 1
        while item[0] < item[1] and text[item[0]] in strip_list:
            item[0] = item[0] + 1
        while item[0] < item[1] and text[item[0]] in opposite and opposite[text[item[0]]] not in text[item[0]:item[1]]:
            item[0] = item[0] + 1
        while item[0] < item[1] and text[item[1] - 1] in strip_list:
            item[1] = item[1] - 1
        while item[0] < item[1] and text[item[1] - 1] in opposite and opposite[text[item[1] - 1]] not in text[item[0]:item[1]]:
            item[1] = item[1] - 1

        # ignore certain words in certain dataset
        if 'medmentions' in dataset.lower() and 'patient' in text[item[0]:item[1]].lower():
            continue
        new_predictions.append(item)

    # Merge same intervals
    sorted_predictions = sorted(new_predictions, key = lambda x: x[0] * POS_CONST + x[1])
    new_predictions = []
    for item in sorted_predictions:
        if new_predictions != []:
            if item[0] == new_predictions[-1][0] and item[1] == new_predictions[-1][1]:
                new_predictions[-1][2] += item[2]
                continue
        new_predictions.append(item)

    # Filter predictions based on threshold
    predictions = new_predictions
    new_predictions = []
    for item in predictions:
        if item[2] > score_threshold:
            new_predictions.append(item)

    return new_predictions

def check_gpu_memory(target_gpu):
    import nvidia_smi
    nvidia_smi.nvmlInit()
    ngpu = nvidia_smi.nvmlDeviceGetCount()
    if ngpu > 0:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(target_gpu)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return info.used

def get_bert_base_params(name):

    medium_names = ['bert-base-multilingual-cased-ner-hrl','bert-base-NER','bert-italian-cased-ner','bert-italian-finetuned-ner','bioBIT',
                    'bert-spanish-cased-finetuned-ner','BioLinkBERT-base','bsc-bio-ehr-es','roberta-base-biomedical-clinical-es',
                    'roberta-es-clinical-trials-ner','NuNER-multilingual-v0.1','BiomedBERT-base-uncased-abstract-fulltext',
                    'BiomedNLP-BiomedBERT-base-uncased-abstract']
    special_names = ['NuNER-v2.0', 'mdeberta-v3-base']
    large_names = ['xlm-roberta-large-english-clinical','xlm-roberta-large-spanish-clinical']

    if name == 'ModernBERT-large':
        cxt_len = 4096
        embed_size = 1024
        batch_size = 1
        grad_acc_steps = 8
    elif name == 'ModernBERT-base':
        cxt_len = 4096
        embed_size = 768
        batch_size = 4
        grad_acc_steps = 2
    elif name in large_names:
        cxt_len = 512
        embed_size = 1024
        batch_size = 8
        grad_acc_steps = 2
    elif name in medium_names:
        cxt_len = 512
        embed_size = 768
        batch_size = 16
        grad_acc_steps = 1
    elif name in special_names:
        cxt_len = 512
        embed_size = 768
        batch_size = 4
        grad_acc_steps = 4
    else:
        raise TypeError('Unknown BERT model!')

    return cxt_len, embed_size, batch_size, grad_acc_steps

def get_special_token_ids(tokenizer, model_type, model_name):
    if model_type == 'bert':
        if model_name in ['NuNER-v2.0','xlm-roberta-large-english-clinical', 'xlm-roberta-large-spanish-clinical']:
            pad_id = tokenizer('<pad>', add_special_tokens=False).data['input_ids']
            sep_id = tokenizer('</s>', add_special_tokens=False).data['input_ids']
            unk_id = tokenizer('<unk>', add_special_tokens=False).data['input_ids']
            mask_id = tokenizer('<mask>', add_special_tokens=False).data['input_ids']
        else:
            pad_id = tokenizer('[PAD]', add_special_tokens=False).data['input_ids']
            sep_id = tokenizer('[SEP]', add_special_tokens=False).data['input_ids']
            unk_id = tokenizer('[UNK]', add_special_tokens=False).data['input_ids']
            mask_id = tokenizer('[MASK]', add_special_tokens=False).data['input_ids']
    else:
        pad_id = tokenizer('<|finetune_right_pad_id|>', add_special_tokens=False).data['input_ids']
        sep_id = tokenizer('\n', add_special_tokens=False).data['input_ids']
        unk_id = tokenizer('<|reserved_special_token_1|>', add_special_tokens=False).data['input_ids']
        mask_id = tokenizer('<|reserved_special_token_2|>', add_special_tokens=False).data['input_ids']

    assert len(pad_id) == 1 and len(sep_id) == 1 and len(unk_id) == 1 and len(mask_id) == 1
    return pad_id[0], sep_id[0], unk_id[0], mask_id[0]

# Convert bio predictions to text intervals
# Input: [[tag, score, token_start_pos]], One more empty token should be added at the end
# Output: [[start, end, score]]
def bio2brat(tags, use_bioe):
    tags = sorted(tags, key=lambda x: x[2])
    result_annos = []
    last_j = None
    score = 0
    score_cnt = 0
    for j in range(len(tags)):
        if last_j is not None:
            end = None
            if use_bioe and tags[j][0] == 3:
                end = j + 1
            elif tags[j][0] in [0, 2]:
                end = j
            if end is not None and end < len(tags):
                result_annos.append([tags[last_j][2], tags[end][2], score / score_cnt])
                score = 0
                score_cnt = 0
                last_j = None
        if tags[j][0] == 2:
            last_j = j
            score += tags[j][1]
            score_cnt += 1
        elif tags[j][0] == 1:
            score += tags[j][1]
            score_cnt += 1

    return result_annos