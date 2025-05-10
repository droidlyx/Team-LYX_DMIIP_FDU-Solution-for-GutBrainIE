import pandas as pd

def convert_test(text, out_f):
    data = {}
    for line in text:
        if '|t|' in line:
            id = line.split('|t|')[0]
            if id not in data:
                data[id] = {'entities': [], 'relations': []}
            data[id]['title'] = line.split('|t|')[1]

        elif '|a|' in line:
            id = line.split('|a|')[0]
            if id not in data:
                data[id] = {'entities': []}
            data[id]['abstract'] = line.split('|a|')[1]

    for id in data:
        title = data[id]['title'].replace('\n', ' ')
        abstract = data[id]['abstract'].replace('\n', ' ')
        out_f.write(str(id) + '|t|' + title + '\n')
        out_f.write(str(id) + '|a|' + abstract + '\n')
        out_f.write('\n')

def convert(text, entities, relations, out_f):
    data = {}
    for line in text:
        if '|t|' in line:
            id = line.split('|t|')[0]
            if id not in data:
                data[id] = {'entities': [], 'relations': []}
            data[id]['title'] = line.split('|t|')[1]

        elif '|a|' in line:
            id = line.split('|a|')[0]
            if id not in data:
                data[id] = {'entities': []}
            data[id]['abstract'] = line.split('|a|')[1]
    for line in entities[1:]:
        items = line.split('\t')
        if len(items) == 7:
            id = items[0]
            data[id]['entities'].append(items[2:])
    for line in relations[1:]:
        items = line.split('\t')
        if len(items) == 13:
            id = items[0]
            re = []
            re.append(items[7])
            find = 0
            for i in range(len(data[id]['entities'])):
                if data[id]['entities'][i][0] == items[2] and data[id]['entities'][i][1] == items[3]:
                    find += 1
                    re.append(i)
                    break
            for i in range(len(data[id]['entities'])):
                if data[id]['entities'][i][0] == items[8] and data[id]['entities'][i][1] == items[9]:
                    find += 1
                    re.append(i)
                    break
            assert find == 2
            data[id]['relations'].append(re)

    for id in data:
        title = data[id]['title'].replace('\n', ' ')
        abstract = data[id]['abstract'].replace('\n', ' ')
        out_f.write(str(id) + '|t|' + title + '\n')
        out_f.write(str(id) + '|a|' + abstract + '\n')
        for anno in data[id]['entities']:
            anno[0] = int(anno[0])
            anno[1] = int(anno[1])
            if anno[2] == 'abstract':
                anno[0] += len(title) + 1
                anno[1] += len(title) + 1
            out_f.write(str(id) + '\t' + str(anno[0]) + '\t' + str(anno[1]) + '\t' + anno[3] + '\t' + anno[4].strip('\n') + '\n')
        for re in data[id]['relations']:
            out_f.write(str(id) + '\trelation\t' + re[0] + '\t' + str(re[1]) + '\t' + str(re[2]) + '\n')
        out_f.write('\n')

if __name__ == '__main__':
    for quality in ['bronze', 'silver', 'gold', 'platinum']:
        text = open('./data/train/Articles/txt_format/articles_train_{}.txt'.format(quality), 'r', encoding='utf-8').readlines()
        entities = open('./data/train/Annotations/Train/{}_quality/txt_format/train_{}_entities.txt'.format(quality, quality), 'r', encoding='utf-8').readlines()
        relations = open('./data/train/Annotations/Train/{}_quality/txt_format/train_{}_relations.txt'.format(quality, quality), 'r', encoding='utf-8').readlines()
        convert(text, entities, relations, open('./data/train/Pubtator/train_set_{}.pubtator'.format(quality), 'w', encoding='utf-8'))

    text = open('./data/train/Articles/txt_format/articles_dev.txt'.format(quality), 'r', encoding='utf-8').readlines()
    entities = open('./data/train/Annotations/Dev/txt_format/dev_entities.txt'.format(quality, quality), 'r', encoding='utf-8').readlines()
    relations = open('./data/train/Annotations/Dev/txt_format/dev_relations.txt'.format(quality, quality), 'r', encoding='utf-8').readlines()
    convert(text, entities, relations, open('./data/eval/dev_set.pubtator'.format(quality), 'w', encoding='utf-8'))

    text = open('./data/test/articles_test.txt', 'r', encoding='utf-8').readlines()
    convert_test(text, open('./data/test/test_set.pubtator', 'w', encoding='utf-8'))

    all_lines = []
    files = ['./data/train/Pubtator/train_set_{}.pubtator'.format(quality) for quality in ['silver', 'gold', 'platinum']]
    files.append('./data/eval/dev_set.pubtator')
    for file in files:
        lines = open(file, 'r', encoding='utf-8'). readlines()
        all_lines += lines

    merged = open('./data/train/Pubtator/train_dev_set.pubtator', 'w', encoding='utf-8')
    merged.writelines(all_lines)

