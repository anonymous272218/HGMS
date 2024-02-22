import pathlib
import yaml
from attrdict import AttrDict
import glob
import os
import torch
import json
from multiprocessing import Pool
from label import label_article
import re
import pickle

REMAP = {"-LRB-": "(", "-RRB-": ")", "-LCB-": "{", "-RCB-": "}",
         "-LSB-": "[", "-RSB-": "]", "&nbsp;": " "}

pattern = re.compile(r"-LRB-|-RRB-|-LCB-|-RCB-|-LSB-|-RSB-|&nbsp;")
corpus_data_map = {'valid': ['valid_data'], 'test': ['test_data'], 'train': [f'data{i}' for i in range(1, 21)]}


def clean(x):
    return pattern.sub(lambda m: REMAP.get(m.group()), x)


def article_to_json(lines):
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    title = []
    text = []
    summary = []
    cur = []
    for line in lines:
        if line == '@title':
            cur = title
        elif line == '@body':
            cur = text
        elif line == '@summary':
            cur = summary
        else:
            cur.append(line)

    text = [clean(sent) for sent in text if 'down for video' not in sent and 'DOWN FOR VIDEO' not in sent]
    summary = [clean(sent) for sent in summary]

    return {'text': text, 'summary': summary}


def get_hash2img_by_corpus_type(config, corpus_type):
    hash2img = {}
    for chunk in corpus_data_map[corpus_type]:
        img_files = glob.glob((config.dataset_dir / chunk / 'img' / "*.jpg").as_posix())
        for file_path in img_files:
            filename = os.path.basename(file_path)
            parts = os.path.splitext(filename)[0].split('_')
            hash = parts[0]
            seq = int(parts[1])
            item = (file_path, seq)
            if hash2img.get(hash):
                hash2img[hash].append(item)
            else:
                hash2img[hash] = [item]
    hash2img_sorted = {}
    for key, value in hash2img.items():
        hash2img_sorted[key] = [item[0] for item in sorted(value, key=lambda x: x[1])]
    return hash2img_sorted


def get_hash2caption_by_corpus_type(config, corpus_type):
    hash2caption = {}
    for chunk in corpus_data_map[corpus_type]:
        caption_files = glob.glob((config.dataset_dir / chunk / 'caption' / "*.caption").as_posix())
        for file_path in caption_files:
            filename = os.path.basename(file_path)
            parts = os.path.splitext(filename)[0].split('_')
            hash = parts[0]
            seq = int(parts[1])
            item = (file_path, seq)
            if hash2caption.get(hash):
                hash2caption[hash].append(item)
            else:
                hash2caption[hash] = [item]
    hash2caption_sorted = {}
    for key, value in hash2caption.items():
        caption_paths = [item[0] for item in sorted(value, key=lambda x: x[1])]
        captions = []
        for path in caption_paths:
            with open(path, 'r', encoding='utf-8') as f:
                captions.append(f.readline().strip())
        hash2caption_sorted[key] = captions
    return hash2caption_sorted


def process_chunk(config):
    for corpus_type in ['valid', 'test', 'train']:
        articles = []
        for chunk in corpus_data_map[corpus_type]:
            article_files = glob.glob((config.dataset_dir / chunk / 'article' / "*.txt").as_posix())
            for file_path in article_files:
                with open(file_path, "r", encoding='utf-8') as file:
                    filename = os.path.basename(file_path)
                    hash = os.path.splitext(filename)[0]
                    item = article_to_json(file.readlines())
                    item['hash'] = hash
                    articles.append(item)

        output = []
        hash2caption = get_hash2caption_by_corpus_type(config, corpus_type)
        for article in articles:
            hash = article['hash']
            caption = hash2caption.get(hash, [])
            article['caption'] = [clean(sent) for sent in caption]
            output.append(json.dumps(article))

        with open((config.save_dir / f'{corpus_type}.raw.jsonl').as_posix(), 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as jsonl_file:
        lines = jsonl_file.readlines()
    output = []
    for line in lines:
        output.append(json.loads(line))
    return output


def label_jsonl(config):
    for corpus_type in ['valid', 'test', 'train']:
        label_file = open((config.save_dir / f'{corpus_type}.label.origin.jsonl').as_posix(), 'w', encoding='utf-8')
        articles = read_jsonl((config.save_dir / f'{corpus_type}.jsonl').as_posix())

        n_cpu = config.n_cpu
        with Pool(n_cpu) as pool:
            labeled_articles = pool.map(label_article, articles)
        label_file.write('\n'.join(labeled_articles))
        label_file.close()


def filter_jsonl(config):
    for corpus_type in ['valid', 'test', 'train']:
        filtered_file = open((config.save_dir / f'{corpus_type}.label.jsonl').as_posix(), 'w', encoding='utf-8')
        labeled_articles = read_jsonl((config.save_dir / f'{corpus_type}.label.origin.jsonl').as_posix())
        filtered_articles = [json.dumps(article) for article in labeled_articles if
                             len(article['text']) > 3 and len(article['summary']) > 0 and
                             len(article['label']) > 0 and len(article['caption']) > 0]

        filtered_file.write('\n'.join(filtered_articles))
        filtered_file.close()


def process_text_and_img_by_blip(config):
    from blip import Blip, BlipDataset, collate_fn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model = Blip.from_pretrained(config.blip_model_path).to(config.device)
    for param in model.parameters():
        param.requires_grad = False

    for corpus_type in ['valid', 'test', 'train']:
        output_dir = pathlib.Path(f'{config.blip_output}/{corpus_type}')
        output_dir.mkdir(exist_ok=True)
        finished_hashes = set([file.stem for file in output_dir.glob('*')])
        hash2img = get_hash2img_by_corpus_type(config, corpus_type)
        labeled_articles = read_jsonl((config.save_dir / f'{corpus_type}.label.jsonl').as_posix())
        labeled_articles = [article for article in labeled_articles if article['hash'] not in finished_hashes]
        dataset = BlipDataset(labeled_articles, hash2img, config.blip_model_path)
        dataloader = DataLoader(dataset, num_workers=4, collate_fn=collate_fn, shuffle=False)
        for hashes, inputs in tqdm(dataloader, delay=0.01, desc=f'Creating {corpus_type} cache'):
            for i in range(len(hashes)):
                hash = hashes[i]
                input = inputs[i].to(config.device)
                output = model(**input)
                itm_score = torch.nn.functional.softmax(output.itm_score, dim=-1)[:, :, 1].cpu()
                image_feature = output.vision_feature.cpu()
                with open(f'{config.blip_output}/{corpus_type}/{hash}.pkl', 'wb') as file:
                    pickle.dump({'image_feature': image_feature, 'itm_score': itm_score}, file)


def process_text_by_bert(config):
    from tqdm import tqdm
    from transformers import RobertaTokenizer, RobertaModel

    bert_model_path = config.bert_model_path
    tokenizer = RobertaTokenizer.from_pretrained(bert_model_path)
    model = RobertaModel.from_pretrained(bert_model_path).to(config.device)
    for param in model.parameters():
        param.requires_grad = False
    for corpus_type in ['valid', 'test', 'train']:
        output_dir = pathlib.Path(f'{config.bert_output}/{corpus_type}')
        output_dir.mkdir(exist_ok=True)
        finished_hashes = set([file.stem for file in output_dir.glob('*')])
        labeled_articles = read_jsonl((config.save_dir / f'{corpus_type}.label.jsonl').as_posix())
        labeled_articles = [article for article in labeled_articles if article['hash'] not in finished_hashes]
        for article in tqdm(labeled_articles, delay=0.01, desc=f'Creating {corpus_type} cache'):
            hash = article['hash']
            text = article['text']
            encoded_input = tokenizer(text[:50], return_tensors='pt', max_length=128, truncation=True,
                                      padding=True).to(config.device)
            output = model(**encoded_input)
            text_feature = output.last_hidden_state[:, 0].cpu()
            torch.save(text_feature, f'{config.bert_output}/{corpus_type}/{hash}.pt')


def process_image_refer(config):
    image_annotation_file = open(config.dataset_dir / 'image_annotation.txt', 'r', encoding='utf-8')
    lines = image_annotation_file.readlines()
    hash2imgref = {}
    for line in lines:
        line = line.strip()
        if line == 'None':
            continue

        for item in line.split():
            filename = os.path.basename(item)
            parts = os.path.splitext(filename)[0].split('_')
            hash = parts[0]
            seq = int(parts[1])
            if hash2imgref.get(hash):
                hash2imgref[hash].append(seq)
            else:
                hash2imgref[hash] = [seq]

    with open(f'{config.save_dir}/hash2imgref.pkl', 'wb') as file:
        pickle.dump(hash2imgref, file)


def label_img_by_blip(config):
    from blip import Blip, BlipSumDataset, collate_fn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model = Blip.from_pretrained(config.blip_model_path).to(config.device)
    for param in model.parameters():
        param.requires_grad = False

    for corpus_type in ['valid', 'test', 'train']:
        hash2img = get_hash2img_by_corpus_type(config, corpus_type)
        labeled_articles = read_jsonl((config.save_dir / f'{corpus_type}.label.jsonl').as_posix())
        dataset = BlipSumDataset(labeled_articles, hash2img, config.blip_model_path)
        dataloader = DataLoader(dataset, num_workers=4, collate_fn=collate_fn, shuffle=False)
        hash2itmlabel = {}
        for hashes, inputs in tqdm(dataloader, delay=0.01, desc=f'Creating {corpus_type} cache'):
            for i in range(len(hashes)):
                hash = hashes[i]
                input = inputs[i].to(config.device)
                output = model(**input)
                itm_score = torch.nn.functional.softmax(output.itm_score, dim=-1)[:, :, 1].cpu()
                hash2itmlabel[hash] = itm_score.squeeze(-1)
        with open(f'{config.save_dir}/{corpus_type}.hash2imglabel.pkl', 'wb') as file:
            pickle.dump(hash2itmlabel, file)


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config['save_dir'] = pathlib.Path(config['save_dir'])
        config['dataset_dir'] = pathlib.Path(config['dataset_dir'])
        config['blip_output'] = pathlib.Path(config['blip_output'])
        config['bert_output'] = pathlib.Path(config['bert_output'])
        config = AttrDict(config)
    config.save_dir.mkdir(exist_ok=True)
    config.blip_output.mkdir(exist_ok=True)
    config.bert_output.mkdir(exist_ok=True)

    # process_chunk(config)
    # label_jsonl(config)
    # filter_jsonl(config)
    # process_text_and_img_by_blip(config)
    # process_image_refer(config)
    # process_text_by_bert(config)
    # label_img_by_blip(config)