import pandas as pd
import json
import argparse

def read_pub(file):
    f = open(file, 'r', encoding='utf-8')
    content = f.readlines()
    f.close()

    all_texts = []
    all_annos = []
    all_lines = []
    all_ids = []
    all_res = []
    pos = 0
    while pos < len(content):
        if '|t|' in content[pos]:
            all_ids.append(content[pos].split('|')[0])
            all_lines.append(content[pos] + content[pos + 1])
            title_len = len(content[pos].split('|t|')[-1])
            text = content[pos].split('|t|')[-1] + content[pos+1].split('|a|')[-1].strip('\n')
            all_texts.append(text)

            pos += 2
            annotations = []
            relations = []
            while pos < len(content) and content[pos] != '\n':
                items = content[pos].split('\t')
                if items[1] == 'relation':
                    relations.append((int(items[3]), int(items[4]), items[2]))
                else:
                    start = int(items[1])
                    end = int(items[2])
                    part = 'title'
                    if start >= title_len:
                        start -= title_len
                        end -= title_len
                        part = 'abstract'
                    annotations.append([start, end, items[3], items[4].strip('\n'), part])
                pos += 1
            all_annos.append(annotations)
            all_res.append(relations)
        else:
            pos += 1

    return all_ids, all_lines, all_texts, all_annos, all_res

def parse_arguments():
    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help='',
                        default='./results/0510091743_merged/predictions/')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    input_path = args.input_path

    NER_result_file = open(input_path + 'LYX_DMIIP_FDU_T61_EnsembleBERT.json', 'w', encoding='utf-8')
    RE1_result_file = open(input_path + 'LYX_DMIIP_FDU_T621_BioLinkBERT.json', 'w', encoding='utf-8')
    RE2_result_file = open(input_path + 'LYX_DMIIP_FDU_T622_BioLinkBERT.json', 'w', encoding='utf-8')
    RE3_result_file = open(input_path + 'LYX_DMIIP_FDU_T623_BioLinkBERT.json', 'w', encoding='utf-8')

    all_ids, _, _, all_annos, all_res = read_pub(input_path + 'final.pubtator')
    NERdict = {}
    REdict1 = {}
    REdict2 = {}
    REdict3 = {}
    for id, annos, relations in zip(all_ids, all_annos, all_res):
        anno_list = []
        sorted_annos = sorted(annos, key=lambda x: x[0] + 1000000 if x[4] == 'abstract' else x[0])
        for anno in sorted_annos:
            anno_list.append({'start_idx': anno[0], 'end_idx': anno[1], 'location': anno[4], 'text_span': anno[2], 'label': anno[3]})
        NERdict[id] = {"entities": anno_list}
        rel2_list = []
        rel3_list = []
        re_list = []
        for rel in relations:
            stype = annos[rel[0]][3]
            ttype = annos[rel[1]][3]
            re_list.append({"subject_text_span": annos[rel[0]][2], 'subject_label': stype, "predicate": rel[2], "object_text_span": annos[rel[1]][2], 'object_label': ttype})
            if (stype, ttype) not in rel2_list:
                rel2_list.append((stype, ttype))
            if (stype, ttype, rel[2]) not in rel3_list:
                rel3_list.append((stype, ttype, rel[2]))
        temp = []
        for item in rel2_list:
            temp.append({'subject_label': item[0], 'object_label': item[1]})
        REdict1[id] = {"binary_tag_based_relations": temp}
        temp = []
        for item in rel3_list:
            temp.append({'subject_label': item[0], "predicate": item[2], 'object_label': item[1]})
        REdict2[id] = {"ternary_tag_based_relations": temp}
        REdict3[id] = {"ternary_mention_based_relations": re_list}

    NER_result_file.write(json.dumps(NERdict, indent=4))
    RE1_result_file.write(json.dumps(REdict1, indent=4))
    RE2_result_file.write(json.dumps(REdict2, indent=4))
    RE3_result_file.write(json.dumps(REdict3, indent=4))

