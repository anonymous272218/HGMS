import datetime
import json
import os
import re
import shutil
import time
import torch
from tqdm import tqdm
from data.cache import GraphDataSet, graph_collate_fn
from model import GraphExt
from tools.config import config
from tools.logger import logger
from tools.tester import ExtTester
from tools.utils import transfer_data, calc_rouge

if config.strict_memory:
    import resource

    memory_limit_bytes = config.memory_limit * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))

_ROUGE_PATH = "/home/xxx/ROUGE-1.5.5"
_PYROUGE_TEMP_FILE = "/home/xxx/ROUGE-1.5.5/temp"

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def load_test_model(model, model_name, eval_dir, save_root):
    """ choose which model will be loaded for evaluation """
    if model_name.startswith('eval'):
        bestmodel_load_path = eval_dir / model_name[4:]
    elif model_name.startswith('train'):
        train_dir = save_root / "train"
        bestmodel_load_path = train_dir / model_name[5:]
    elif model_name == "earlystop":
        train_dir = save_root / "train"
        bestmodel_load_path = train_dir / 'earlystop'
    else:
        logger.error("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
    if not bestmodel_load_path.exists():
        logger.error("Restoring %s for testing...The path %s does not exist!", model_name, bestmodel_load_path)
        return None
    logger.info("[INFO] Restoring %s for testing...The path is %s", model_name, bestmodel_load_path)

    model.load_state_dict(torch.load(bestmodel_load_path, map_location=config.device))

    return model


def clean(x):
    x = x.lower()
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def pyrouge_score_all(hyps_list, refer_list, remap=True):
    from pyrouge import Rouge155
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join(_PYROUGE_TEMP_FILE, nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT, 'result')
    MODEL_PATH = os.path.join(PYROUGE_ROOT, 'gold')
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)
    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        model_file = os.path.join(MODEL_PATH, 'Reference.A.%d.txt' % i)

        refer = clean(refer_list[i]) if remap else refer_list[i]
        hyps = clean(hyps_list[i]) if remap else hyps_list[i]

        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))
        with open(model_file, 'wb') as f:
            f.write(refer.encode('utf-8'))

    r = Rouge155(_ROUGE_PATH, log_level='WARNING')

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Reference.[A-Z].#ID#.txt'

    try:
        output = r.convert_and_evaluate(rouge_args="-e %s -a -m -n 2 -d" % os.path.join(_ROUGE_PATH, "data"))
        output_dict = r.output_to_dict(output)
    finally:
        shutil.rmtree(PYROUGE_ROOT)

    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = output_dict['rouge_1_precision'], \
        output_dict['rouge_1_recall'], output_dict['rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = output_dict['rouge_2_precision'], \
        output_dict['rouge_2_recall'], output_dict['rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = output_dict['rouge_l_precision'], \
        output_dict['rouge_l_recall'], output_dict['rouge_l_f_score']
    return scores


def show_rouge(scores_all, method_name):
    method_str = '\n================%s================' % method_name
    res = "\nRouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
          + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
          + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(method_str + res)


def run_test(model, dataset, loader, model_name):
    test_dir = config.save_root / "test"  # make a subdir of the root dir for eval data
    eval_dir = config.save_root / "eval"
    test_dir.mkdir(exist_ok=True)
    if not eval_dir.exists():
        logger.exception("eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception("eval_dir %s doesn't exist. Run in train mode to create it." % (eval_dir))

    model = load_test_model(model, model_name, eval_dir, config.save_root)
    if model is None:
        return
    model.eval()

    iter_start_time = time.time()
    with torch.no_grad():
        tester = ExtTester(model, config.m, limited=config.limited, test_dir=test_dir)
        pbar = tqdm(loader, delay=0.01, desc='test')
        for i, (data, origin) in enumerate(pbar):
            data = transfer_data(data, config.device)
            tester.evaluation(data, origin, config.blocking, config.threshold)

    running_avg_loss = tester.running_avg_loss
    logger.info("The number of pairs is %d", tester.rougePairNum)
    logger.info("The threshold is %f", config.threshold)

    if config.model == 'ext':
        if config.save_label:
            # save label and do not calculate rouge
            now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            label_file_path = test_dir / ('label_' + now)
            with open(label_file_path, "w") as label_file:
                json.dump(tester.extractLabel, label_file)
            logger.info("Write the Evaluation into %s", label_file_path)
            tester.SaveDecodeFile()
            logger.info('   | end of test | time: {:5.2f}s | '.format((time.time() - iter_start_time)))
            return

        if config.use_pyrouge:
            scores_all = pyrouge_score_all(tester.hyps, tester.refer)
            show_rouge(scores_all, 'pyrouge')
        else:
            rouge1, rouge2, rougeLsum = calc_rouge(tester.hyps, tester.refer)
            logger.info("\n================rouge_score================\nrouge1: %.6f \nrouge2: %.6f \nrougeL: %.6f"
                        % (rouge1, rouge2, rougeLsum))
        tester.getMetric()
        tester.SaveDecodeFile()

    logger.info('[INFO] End of test | time: {:5.2f}s | test loss {:5.4f} | '.format((time.time() - iter_start_time),
                                                                                    float(running_avg_loss)))


def main():
    torch.set_printoptions(threshold=50000)

    logger.info("Pytorch %s", torch.__version__)
    logger.info(config)

    model = GraphExt()

    model = model.to(config.device)
    test_dataset = GraphDataSet(config.datasets_dir, 'test', disperse=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                              num_workers=config.num_workers, collate_fn=graph_collate_fn,
                                              pin_memory=True)

    if config.test_model == "multi":
        run_test(model, test_dataset, test_loader, 'evalbestFmodel')
        for i in range(3):
            model_name = "evalbestmodel_%d" % i
            run_test(model, test_dataset, test_loader, model_name)
        run_test(model, test_dataset, test_loader, 'trainbestmodel')
        run_test(model, test_dataset, test_loader, 'earlystop')
    else:
        run_test(model, test_dataset, test_loader, config.test_model)


if __name__ == '__main__':
    main()
