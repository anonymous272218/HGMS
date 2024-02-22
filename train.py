import sys
import time
import torch
import argparse
from tqdm import tqdm
from tools.tester import ExtTester
from model import GraphExt
from tools.logger import logger
from module.vocabulary import FileVocab, DictVocab
from module.embedding import Word_Embedding
from data.cache import GraphDataSet, create_cache, graph_collate_fn
from tools.utils import save_json, read_json, plot_loss_curve, transfer_data, init_seeds
from tools.config import config
from torch.utils.data import DataLoader, distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tools.utils import calc_ext_loss, calc_rouge

# DDP
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int, default=-1)
LOCAL_RANK = parser.parse_args().local_rank
USE_DDP = LOCAL_RANK != -1
IS_MASTER_NODE = LOCAL_RANK in {-1, 0}
embedding_freeze_mask = None

if USE_DDP:
    torch.cuda.set_device(LOCAL_RANK)
    config.device = torch.device('cuda', LOCAL_RANK)
    dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

if config.strict_memory:
    import resource

    memory_limit_bytes = config.memory_limit * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_RSS, (memory_limit_bytes, memory_limit_bytes))


def save_model(model, save_file):
    if USE_DDP:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    with open(save_file, 'wb') as f:
        torch.save(state_dict, f)
    logger.info('Saving model to %s', save_file)


def setup_training(model, train_loader):
    records = {
        'epoch': -1,
        'best_train_loss': float('inf'),
        'best_valid_loss': float('inf'),
        'top_valid_loss': [float('inf'), float('inf'), float('inf')],
        'best_valid_F': 0,
        'non_descent_cnt': 0,
        'train_loss_history': [],
        'valid_loss_history': []
    }
    train_dir = config.save_root / "train"
    if (train_dir / 'bestmodel').exists() and not config.ignore_save_folder:
        if config.restore_model != 'None':
            logger.info("Resuming training from %s", config.restore_model)
            restore_model_file = train_dir / config.restore_model
            model.load_state_dict(torch.load(restore_model_file, map_location=config.device))
            records = read_json(train_dir / 'record.json')
            logger.info('loaded training record: %s', records)
        else:
            logger.error("The save folder is not empty")
            sys.exit(1)
    else:
        train_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Create new model for training...")

    if USE_DDP:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    try:
        run_training(model, train_loader, train_dir, records)
    except KeyboardInterrupt:
        logger.error("Caught keyboard interrupt on worker. Stopping supervisor...")
        if IS_MASTER_NODE:
            save_model(model, train_dir / "earlystop")
        sys.exit(1)


def lr_lambda(epoch):
    decay_factors = [1, 0.65, 0.5, 0.4, 0.3, 0.2]
    return decay_factors[min(epoch // 1, len(decay_factors) - 1)]


def freeze_weight(model):
    if USE_DDP:
        model.module.encoder.word_embed.weight.grad[embedding_freeze_mask] = 0
    else:
        model.encoder.word_embed.weight.grad[embedding_freeze_mask] = 0


def run_training(model, train_loader, train_dir, records):
    logger.info("Starting run_training")
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    # optimizer = torch.optim.Adam(
    #     [{'params': model.parameters(), 'initial_lr': config.lr, 'lr': config.lr}], lr=config.lr)

    group1 = []
    group2 = []
    for name, param in model.named_parameters():
        if 'bert_model' in name:
            group1.append(param)
        else:
            group2.append(param)
    optimizer = torch.optim.Adam(
        [
            {'params': group1, 'initial_lr': config.lr / 10, 'lr': config.lr / 10},
            {'params': group2, 'initial_lr': config.lr, 'lr': config.lr},
        ], lr=config.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=records['epoch'], verbose=False)
    for epoch in range(records['epoch'] + 1, config.n_epochs):
        logger.info(f'==============================Epoch {epoch}==============================')
        for i, lr in enumerate(scheduler.get_last_lr()):
            logger.info(f'LR Group {i + 1}: {lr}')
        if USE_DDP:
            train_loader.sampler.set_epoch(epoch)
        records['epoch'] = epoch
        epoch_s_loss = 0.0
        epoch_i_loss = 0.0
        epoch_start_time = time.time()

        model.train()

        pbar = train_loader
        if IS_MASTER_NODE:
            pbar = tqdm(train_loader, desc='train', postfix={'avg_loss': '0.0', 'last_loss': '0.0'})

        for i, (data, _) in enumerate(pbar):
            data = transfer_data(data, config.device)
            sent_score, img_score = model.forward(data)
            s_loss = calc_ext_loss(sent_score, data['s_label'], criterion)
            i_loss = calc_ext_loss(img_score, data['i_label'], criterion)
            loss = config.loss_rate * s_loss + (1 - config.loss_rate) * i_loss
            loss.backward()
            freeze_weight(model)
            optimizer.step()
            optimizer.zero_grad()
            f_s_loss = float(s_loss)
            f_i_loss = float(i_loss)
            epoch_s_loss += f_s_loss
            epoch_i_loss += f_i_loss
            if IS_MASTER_NODE:
                pbar.set_postfix(
                    {'avg_s_loss': f'{epoch_s_loss / (i + 1):.4f}', 'avg_i_loss': f'{epoch_i_loss / (i + 1):.4f}',
                     's_loss': f'{f_s_loss:.4f}', 'i_loss': f'{f_i_loss:.4f}'})
                pbar.update()
            # end batch ------------------------------------------------------------------------------------------------
        scheduler.step()

        if IS_MASTER_NODE:
            avg_s_loss = epoch_s_loss / len(train_loader)
            avg_i_loss = epoch_i_loss / len(train_loader)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | avg_s_loss {:5.4f} | avg_i_loss {:5.4f} | '
                        .format(epoch, (time.time() - epoch_start_time), avg_s_loss, avg_i_loss))

            if avg_s_loss < records['best_train_loss']:
                save_file = train_dir / "bestmodel"
                logger.info('New best model with %.4f -> %.4f train_loss', records['best_train_loss'], avg_s_loss)
                save_model(model, save_file)
                records['best_train_loss'] = avg_s_loss

            run_eval(model, records)

            logger.info(f'Max memory allocated : {(torch.cuda.max_memory_allocated(config.device) / 1024 / 1024):.2f}M')
            # save record
            records['train_loss_history'].append(avg_s_loss)
            logger.info('Save training record: %s', records)
            plot_loss_curve(records['train_loss_history'], records['valid_loss_history'], train_dir / 'loss_curve.png')
            save_json(train_dir / 'record.json', records)

            if records['non_descent_cnt'] >= config.patience:
                logger.error("valid descent exceeds the patience value, stop training !")
                save_model(model, train_dir / "earlystop")
                return


def run_eval(model, records):
    best_loss, best_F, non_descent_cnt = \
        (records['best_valid_loss'], records['best_valid_F'], records['non_descent_cnt'])
    logger.info("Starting eval for this model ...")
    valid_dataset = GraphDataSet(config.datasets_dir, 'valid', disperse=True)
    loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, collate_fn=graph_collate_fn,
                        pin_memory=True)
    eval_dir = config.save_root / "eval"  # make a subdir of the root dir for eval data
    eval_dir.mkdir(exist_ok=True)

    model.eval()

    iter_start_time = time.time()

    with torch.no_grad():
        tester = ExtTester(model, config.m)
        for i, (data, origin) in enumerate(tqdm(loader, desc='valid')):
            data = transfer_data(data, config.device)
            tester.evaluation(data, origin)

    running_avg_loss = tester.running_avg_loss

    if config.model == 'ext':
        rouge1, rouge2, rougeLsum = calc_rouge(tester.hyps, tester.refer)
        logger.info("\n================rouge_score================\nrouge1: %.6f \nrouge2: %.6f \nrougeL: %.6f"
                    % (rouge1, rouge2, rougeLsum))

    logger.info('End of valid | time: {:5.2f}s | valid loss {:5.4f} | '.format((time.time() - iter_start_time),
                                                                               float(running_avg_loss)))

    if config.model == 'ext':
        tester.getMetric()
        F = tester.labelMetric.item()
    else:
        F = best_F

    if running_avg_loss < best_loss:
        logger.info('New best model with %.4f -> %.4f valid_loss', best_loss, running_avg_loss)
        best_loss = running_avg_loss
        non_descent_cnt = 0
    else:
        non_descent_cnt += 1

    if running_avg_loss < max(records['top_valid_loss']):
        index = records['top_valid_loss'].index(max(records['top_valid_loss']))
        records['top_valid_loss'][index] = running_avg_loss
        logger.info('New top valid loss : %.4f', running_avg_loss)
        save_model(model, eval_dir / ('bestmodel_%d' % index))

    if best_F < F:
        logger.info('New best model with %.4f -> %.4f F', best_F, F)
        save_model(model, eval_dir / 'bestFmodel')
        best_F = F

    records['best_valid_loss'] = best_loss
    records['best_valid_F'] = best_F
    records['non_descent_cnt'] = non_descent_cnt
    records['valid_loss_history'].append(running_avg_loss)


def main():
    # set the seed
    init_seeds(config.seed)
    torch.set_printoptions(threshold=50000)

    logger.info("Pytorch %s", torch.__version__)
    logger.info(config)

    # init word_embedding
    vocab = FileVocab(config.datasets_dir / "vocab", config.vocab_size, verbose=True)
    embed_loader = Word_Embedding(config.embedding_file, vocab)
    vectors = embed_loader.load_my_vecs(config.word_emb_dim)
    pretrained_weight, oov_ids = embed_loader.add_unknown_words_by_avg(vectors, config.word_emb_dim)
    global embedding_freeze_mask
    embedding_freeze_mask = torch.ones(len(pretrained_weight), device=config.device, dtype=torch.bool)
    embedding_freeze_mask[oov_ids] = False
    pretrained_weight = torch.Tensor(pretrained_weight)
    padding_idx = 0
    pretrained_weight[padding_idx].fill_(0)
    embedding_freeze_mask[padding_idx] = True

    if IS_MASTER_NODE:
        create_cache(vocab, disperse=True)

    model = GraphExt()
    model = model.to(config.device)
    model.encoder.word_embed.weight.data.copy_(pretrained_weight)

    train_dataset = GraphDataSet(config.datasets_dir, 'train', disperse=True)
    sampler = distributed.DistributedSampler(train_dataset) if USE_DDP else None
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              sampler=sampler, shuffle=not USE_DDP,
                              num_workers=config.num_workers, collate_fn=graph_collate_fn,
                              pin_memory=True)
    setup_training(model, train_loader)


if __name__ == '__main__':
    main()
