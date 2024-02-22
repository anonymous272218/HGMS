import copy
from tqdm import tqdm
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from data.dataset import *


def _create_cache(vocab):
    save_root = config.datasets_dir / ('graph_cache_v_' + config.cache_version)
    sign_file = save_root / 'complete'
    if sign_file.exists() and config.cache_use_exist:
        logger.info('Use the existing cache')
        return
    logger.info('Start creating graph cache...')
    save_root.mkdir(exist_ok=True)
    for corpus_type in ['train', 'valid', 'test']:
        dataset = ExampleSet(config.datasets_dir, corpus_type, vocab, config.img_weight_limit)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.cache_batch, shuffle=False,
                                                 num_workers=config.cache_worker, collate_fn=cache_collate_fn)
        data_list = []
        origin_list = []
        for batch in tqdm(dataloader, delay=0.01, desc=f'Creating {corpus_type} cache'):
            copy_batch = copy.deepcopy(batch)
            del batch
            data_dict = copy_batch[0]
            origin = copy_batch[1]
            data_list.extend(data_dict)
            if corpus_type != 'train':
                origin_list.extend(origin)
            if config.get('dev') and len(data_list) > 100:
                break

        save_info((save_root / (corpus_type + '_info.pkl')).as_posix(), {'data': data_list, 'origin': origin_list})
    sign_file.touch()
    logger.info('Finish creating graph cache')


def _create_cache_disperse(vocab):
    save_root = config.datasets_dir / ('graph_cache_v_' + config.cache_version)
    sign_file = save_root / 'complete'
    if sign_file.exists() and config.cache_use_exist:
        logger.info('Use the existing cache')
        return
    logger.info('Start creating graph cache...')
    save_root.mkdir(exist_ok=True)
    for corpus_type in ['train', 'valid', 'test']:
        dataset = ExampleSet(config.datasets_dir, corpus_type, vocab, config.img_weight_limit)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.cache_batch, shuffle=False,
                                                 num_workers=config.cache_worker, collate_fn=cache_collate_fn)
        data_list = []
        origin_list = []
        index = 0

        def save_data(datas, origins, id_):
            (save_root / corpus_type).mkdir(exist_ok=True)
            for i, data in enumerate(datas):
                save_info(f'{save_root}/{corpus_type}/{id_}.pkl', {'data': (data, origins[i])})
                id_ += 1
            return id_

        for batch in tqdm(dataloader, delay=0.01, desc=f'Creating {corpus_type} cache'):
            copy_batch = copy.deepcopy(batch)
            del batch
            data_dict = copy_batch[0]
            origin = copy_batch[1]
            data_list.extend(data_dict)
            origin_list.extend(origin)
            if config.get('dev') and len(data_list) > 100:
                break
            if len(data_list) > 10000:
                index = save_data(data_list, origin_list, index)
                data_list = []
                origin_list = []

        index = save_data(data_list, origin_list, index)
        with open(save_root / corpus_type / 'length.txt', 'w') as file:
            file.write(str(index))
    sign_file.touch()
    logger.info('Finish creating graph cache')


def create_cache(vocab, disperse=False):
    if disperse:
        return _create_cache_disperse(vocab)
    return _create_cache(vocab)


class GraphDataSet(torch.utils.data.Dataset):
    def __init__(self, datasets_dir, corpus_type, disperse=False):
        self.graph_cache_dir = datasets_dir / f'graph_cache_v_{config.cache_version}'
        self.corpus_type = corpus_type
        self.disperse = disperse
        if disperse:
            self.graph_cache_dir = self.graph_cache_dir / self.corpus_type
            with open(self.graph_cache_dir / 'length.txt', 'r') as file:
                self.length = int(file.readline())
        else:
            info_dict = load_info((self.graph_cache_dir / f'{corpus_type}_info.pkl').as_posix())
            self.data = info_dict['data']
            self.origin = info_dict['origin']
            self.length = len(self.data)

    def __getitem__(self, index):
        if self.disperse:
            return load_info(f'{self.graph_cache_dir}/{index}.pkl')['data']
        data = self.data[index]
        origin = (None if self.corpus_type == 'train' else self.origin[index])

        return data, origin

    def __len__(self):
        return self.length


def cache_collate_fn(batch):
    data_dicts = [e[0] for e in batch]
    origins = [e[1] for e in batch]
    return data_dicts, origins
