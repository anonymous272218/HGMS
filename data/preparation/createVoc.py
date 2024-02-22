import os
import yaml
import json
import nltk
from attrdict import AttrDict


def PrintInformation(keys, allcnt):
    # Vocab > 10
    cnt = 0
    first = 0.0
    for key, val in keys:
        if val >= 10:
            cnt += 1
            first += val
    print("appearance > 10 cnt: %d, percent: %f" % (cnt, first / allcnt))  # 416,303

    # first 30,000, last freq 31
    if len(keys) > 30000:
        first = 0.0
        for k, v in keys[:30000]:
            first += v
        print("First 30,000 percent: %f, last freq %d" % (first / allcnt, keys[30000][1]))

    # first 50,000, last freq 383
    if len(keys) > 50000:
        first = 0.0
        for k, v in keys[:50000]:
            first += v
        print("First 50,000 percent: %f, last freq %d" % (first / allcnt, keys[50000][1]))

    # first 100,000, last freq 107
    if len(keys) > 100000:
        first = 0.0
        for k, v in keys[:100000]:
            first += v
        print("First 100,000 percent: %f, last freq %d" % (first / allcnt, keys[100000][1]))


def main(save_dir, label_dir):
    os.makedirs(save_dir, exist_ok=True)
    saveFile = os.path.join(save_dir, "vocab")
    allword = []
    cnt = 0
    with open(os.path.join(label_dir, 'train.label.jsonl'), encoding='utf8') as f:
        for line in f:
            e = json.loads(line)
            sents = [sent.lower() for sent in e["text"]]
            summaries = [sent.lower() for sent in e["summary"]]
            text = " ".join(sents)
            summary = " ".join(summaries)
            allword.extend(nltk.word_tokenize(text))
            allword.extend(nltk.word_tokenize(summary))
            cnt += 1
    print("Training set of dataset has %d example" % cnt)

    fdist1 = nltk.FreqDist(allword)

    fout = open(saveFile, "w", encoding='utf-8')
    keys = fdist1.most_common()
    for key, val in keys:
        try:
            fout.write("%s\t%d\n" % (key, val))
        except UnicodeEncodeError as e:
            # print(repr(e))
            # print(key, val)
            continue

    fout.close()

    allcnt = fdist1.N()
    allset = fdist1.B()
    print("All appearance %d, unique word %d" % (allcnt, allset))

    PrintInformation(keys, allcnt)


if __name__ == '__main__':
    save_dir = 'G:/datasets/dailymail_processed_v2/cache'
    label_dir = 'G:/datasets/dailymail_processed_v2/'
    main(save_dir, label_dir)
