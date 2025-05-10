import glob
from model.models import MLP_Head
from model.utils import *
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import numpy as np
import os
import requests
import argparse

def read_pub(file):
    f = open(file, 'r', encoding='utf-8')
    content = f.readlines()
    f.close()

    all_texts = []
    all_annos = []
    all_lines = []
    all_ids = []
    pos = 0
    while pos < len(content):
        if '|t|' in content[pos]:
            all_ids.append(content[pos].split('|')[0])
            all_lines.append(content[pos] + content[pos + 1])

            text = content[pos].split('|t|')[-1] + content[pos+1].split('|a|')[-1].strip('\n')
            all_texts.append(text)

            pos += 2
            annotations = []
            while pos < len(content) and content[pos] != '\n':
                items = content[pos].split('\t')
                annotations.append([int(items[1]), int(items[2]), items[3], items[4].strip('\n')])
                pos += 1
            all_annos.append(annotations)
        else:
            pos += 1

    return all_ids, all_lines, all_texts, all_annos

def parse_arguments():
    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument("--re_model_path", help='',
                        default='./finetuned_models/finetuned_classifiers/BioLinkBERT-base/')
    parser.add_argument("--input_path", help='',
                        default='./results/0510091743_merged/')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    val = False
    re_model_path = args.re_model_path
    input_path = args.input_path

    saved = torch.load(re_model_path + 'saved.pt', weights_only=True)
    base_model_path = saved["base_model_path"]
    model_name = base_model_path.strip('/').split('/')[-1]
    train_config = saved["train_config"]
    train_config["hidden_dropout_prob"] = 0
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    pad_id = tokenizer(tokenizer.pad_token, add_special_tokens=False).data['input_ids'][0]

    encoder = AutoModel.from_pretrained(re_model_path).bfloat16().cuda()
    encoder.eval()
    classfier = MLP_Head(train_config, 1024 if model_name == 'xlm-roberta-large-english-clinical' else 768).bfloat16().cuda()
    classfier.load_state_dict(saved['classfier_state_dict'])

    pub_file = glob.glob(input_path + 'predictions/*.pubtator')[0]
    all_data = generate_data(pub_file, tokenizer)

    all_relations = {}
    test_batch_size = 64
    tp = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(all_data), test_batch_size)):
            batch = [x['tokens'] for x in all_data[i:i+test_batch_size]]
            batch = pad(batch, pad_id)
            batch = torch.tensor(batch).cuda()

            mask = torch.ones_like(batch)
            mask[batch == pad_id] = 0
            output_embed = encoder(batch, attention_mask=mask)[0][:, 0, :]
            logits, _, _ = classfier(output_embed)
            preds = torch.argmax(logits, -1)

            for j in range(len(preds)):
                if val:
                    answer = preds[j].item()
                    gt = all_data[i+j]['label']
                    if answer == 1 and gt == 1:
                        tp += 1
                    elif answer == 1:
                        fp += 1
                    elif gt == 1:
                        fn += 1
                elif preds[j].item() == 1:
                    relation = all_data[i+j]['relation']
                    if relation[0] not in all_relations:
                        all_relations[relation[0]] = []
                    all_relations[relation[0]].append((relation[1],relation[2],relation[3]))

        if val:
            precision, recall, f_score = prf_metrics(tp, fp, fn)
            print(f"val precision: {precision:.4f}, val recall: {recall:.4f}, val f_score: {f_score:.4f}")
        else:
            all_ids, all_lines, all_texts, all_annos = read_pub(pub_file)
            file = open(input_path + 'predictions/final.pubtator', "w", encoding='utf-8')
            for i in range(len(all_ids)):
                file.write(all_lines[i])
                for anno in all_annos[i]:
                    file.write(all_ids[i] + '\t' + str(anno[0]) + '\t' + str(anno[1]) + '\t' + anno[2] + '\t' + anno[3] + '\n')
                if all_ids[i] in all_relations:
                    for re in all_relations[all_ids[i]]:
                        file.write(all_ids[i] + '\trelation\t' + re[2] + '\t' + str(re[0]) + '\t' + str(re[1]) + '\n')
                file.write('\n')


