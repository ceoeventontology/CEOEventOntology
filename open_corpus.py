# Created by ceoeventontology at 2023/2/7
import os
os.environ['KMP_WARNINGS'] = '0'
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import argparse
import sys
sys.path.append('./SRL-English')
import json
from tqdm import tqdm
import subprocess
from allennlp.commands import main
import yaml
from open_utilities import convert_tokens, convert_id_to_srl_input, salient_preparation, get_sent_sbert_repre, \
    get_token_repre, extract_event_repre, preserve_events, combine_plain_embeddings, \
    generate_linkage_ward_tree
from name_generation import convert_tree_file_to_nx_tree, generative, combine_event_info_with_name
import logging
logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger.setLevel(logging.DEBUG)
import numpy as np
from copy import deepcopy
from improve_embedding import run_embedding
import pickle
import networkx as nx

def event_extraction(corpus_path, res_folder, gpu_id, overwrite):
    parent_folder = os.path.join(res_folder, 'event_extraction')
    os.makedirs(parent_folder, exist_ok=True)
    logger.info('step I: convert into sentences')
    """
    input: sentences per line 
    output: dict per line
        {"tokens": ["Death", "of", "Arafat", "(", "1", ")", "Controversial", "PLO", "leader", "Yasser", "Arafat", "died", "in", "a", "Paris", "hospital", "last", "week", "."]}
    """
    token_SRL_path = convert_tokens(corpus_path, parent_folder)

    logger.info('step II-I: SRL for verb')
    if not os.path.exists(f"{parent_folder}/verb_sense_srl.txt") or overwrite:
        sys.argv = [
            "allennlp",  # command name, not used by main
            "predict",
            "./models/verb-sense-srl.tar.gz",
            token_SRL_path,
            "--output-file", f"{parent_folder}/verb_sense_srl.txt",
            "--include-package", "verb_sense_srl",
            "--predictor", "sense-semantic-role-labeling",
        ]
        if gpu_id >= 0:
            sys.argv += ["--cuda-device", str(gpu_id)]
        main()

    logger.info('step II-II: SRL for nom')
    logger.info('1. nom identification')
    if not os.path.exists(token_SRL_path+'-id-output.txt') or overwrite:
        sys.argv = [
            "allennlp",  # command name, not used by main
            "predict",
            "./models/nom-id-bert.tar.gz",
            token_SRL_path,
            "--output-file", token_SRL_path+'-id-output.txt',
            "--include-package", "id_nominal",
            "--predictor", "nombank-id",
            # "--cuda-device", str(gpu_id),
        ]
        if gpu_id >= 0:
            sys.argv += ["--cuda-device", str(gpu_id)]
        main()
    logger.info('2. convert id')
    if not os.path.exists(f'{token_SRL_path+"-srl-input.txt"}') or overwrite:
        convert_id_to_srl_input(token_SRL_path+"-id-output.txt", token_SRL_path+"-srl-input.txt")

    logger.info('3. nom srl')
    if not os.path.exists(f"{parent_folder}/nom_sense_srl.txt") or overwrite:
        sys.argv = [
            "allennlp",  # command name, not used by main
            "predict",
            "./models/nom-sense-srl.tar.gz",
            token_SRL_path+"-srl-input.txt",
            "--output-file", f"{parent_folder}/nom_sense_srl.txt",
            "--include-package", "nominal_sense_srl",
            "--predictor", "nombank-sense-srl",
            # "--cuda-device", str(gpu_id),
        ]
        if gpu_id >= 0:
            sys.argv += ["--cuda-device", str(gpu_id)]
        main()
    return

def salient_event_detection(corpus_path, res_folder, top_percentile, gpu_id, seed, overwrite):
    parent_folder = os.path.join(res_folder, 'salient_event_detection')
    os.makedirs(parent_folder, exist_ok=True)
    dataset = corpus_path.split('/')[-1].rstrip('.txt')
    conda_path = os.path.join(os.popen('echo $(conda info --base)').read().rstrip('\n'), 'etc/profile.d/conda.sh')
    conda_env = os.getenv('CONDA_PREFIX')
    assert os.path.exists(conda_path) and os.path.exists(conda_env)

    logger.info('step I: process SRL results')
    nom_doc_path, verb_doc_path = os.path.join(parent_folder, 'nom_sense_srl_doc.txt'), os.path.join(parent_folder, 'verb_sense_srl_doc.txt')
    nom_path = os.path.join(res_folder, 'event_extraction', 'nom_sense_srl.txt')
    verb_path = os.path.join(res_folder, 'event_extraction', 'verb_sense_srl.txt')
    if not os.path.exists(nom_doc_path) or not os.path.exists(verb_doc_path) or overwrite:
        salient_preparation(corpus_path, nom_path, verb_path, nom_doc_path, verb_doc_path)

    logger.info('step II-I: saliency detection - preprocessing')
    suffix = '_doc'
    event_doc_path = os.path.join(parent_folder, 'longformer_saliency', f'sent{suffix}.hf.json')
    if not os.path.exists(event_doc_path) or overwrite:
        os.makedirs(os.path.dirname(event_doc_path), exist_ok=True)
        num_lines_nom, num_lines_verb = sum(1 for i in open(nom_doc_path)), sum(1 for i in open(verb_doc_path))
        assert num_lines_nom == num_lines_verb
        logger.info(f'read {num_lines_nom} lines from {dataset}')
        sample_list = list()
        for line_idx, (nom_line, verb_line) in tqdm(enumerate(zip(open(nom_doc_path), open(verb_doc_path))), total=num_lines_verb):
            nom_info, verb_info = json.loads(nom_line), json.loads(verb_line)
            assert nom_info['words'] == verb_info['words']
            tokens = verb_info['words']
            cur_seq = {
                'id': nom_info['doc_id'],
                'event_index': list(),
                'tokens': tokens,
            }
            for nominal_indice in nom_info['nominal_indices']:
                cur_seq['event_index'].append([nominal_indice[0], nominal_indice[-1]+1])
            for verb_index in verb_info['verb_index']:
                if [verb_index, verb_index+1] not in cur_seq['event_index']:
                    cur_seq['event_index'].append([verb_index, verb_index+1])
            if len(cur_seq['event_index']) == 0:
                logger.info(f'no events discovered in {nom_info["doc_id"]}, skip...')
                continue
            # reorder event_index
            cur_seq['event_index'] = np.array(cur_seq['event_index'])
            order = np.argsort(cur_seq['event_index'][:, 0])
            cur_seq['event_index'] = cur_seq['event_index'][order].tolist()
            sample_list.append(deepcopy(cur_seq))
        logger.info(f'{dataset}-sample_list: {len(sample_list)}')
        fout = open(event_doc_path, 'w')
        for sample in sample_list:
            fout.write(json.dumps(sample) + '\n')
        fout.close()

    logger.info('step II-II: saliency detection')
    output_dir = os.path.join(parent_folder, "longformer_saliency", "nyt")
    if not os.path.exists(os.path.join(output_dir, 'predictions.txt')) or overwrite:
        model_folder = './models/saliency_longformer_nyt'
        cmd = ''
        cmd += 'python saliency_prediction.py --max_seq_length 4096 '
        cmd += f'--test_file \"{event_doc_path}\" '
        cmd += f'--output_dir \"{output_dir}\" '
        cmd += f'--per_device_eval_batch_size 24 --overwrite_output_dir '
        cmd += f'--do_predict --seed {seed} --model_name_or_path {model_folder} '
        if gpu_id >= 0:
            cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} ' + cmd + '--fp16 --fp16_full_eval'
        cmd = f'. {conda_path} && conda activate {conda_env} && ' + cmd
        logger.info(f'cmd: {cmd}')

        p = subprocess.Popen(cmd, shell=True)
        p.wait()

    logger.info('step II-III: saliency detection - map saliency from doc to sent')
    required_sent_id_list = list()
    for line in open(corpus_path):
        raw_info = json.loads(line)
        sent_id = raw_info['sent_id']
        required_sent_id_list.append(sent_id)
    salient_nom_path = os.path.join(parent_folder, 'nyt', 'nom_sense_srl_saliency.txt')
    salient_verb_path = os.path.join(parent_folder, 'nyt', 'verb_sense_srl_saliency.txt')
    if not os.path.exists(salient_nom_path) or overwrite:
        pred_doc_path = os.path.join(parent_folder, 'longformer_saliency', 'nyt', 'predictions.txt')
        pred_doc_res = dict()
        for line in open(pred_doc_path):
            res = json.loads(line)
            pred_doc_res[res['doc_id']] = res

        num_nom_doc, num_verb_doc = sum(1 for i in open(nom_doc_path)), sum(1 for i in open(verb_doc_path))
        assert num_nom_doc == num_verb_doc
        # it's possible that fewer documents in prediction since some of documents have no recognized events
        sent_verb_list, sent_nom_list = list(), list()
        for idx, (nom_line, verb_line) in tqdm(enumerate(zip(open(nom_doc_path), open(verb_doc_path))), total=num_verb_doc):
            nom_info, verb_info = json.loads(nom_line), json.loads(verb_line)
            assert nom_info['doc_id'] == verb_info['doc_id']
            doc_id = nom_info['doc_id']

            sent_len_cumsum = np.cumsum(verb_info['sent_len'])
            verbs_len_cumsum = np.cumsum(verb_info['verbs_len'])
            nominals_len_cumsum = np.cumsum(nom_info['nominals_len'])
            for sent_id in range(len(verb_info['sent_len'])):
                nickname = f'{doc_id}-{sent_id}'
                if nickname not in required_sent_id_list:
                    continue
                if sent_id == 0:
                    word_start_idx = 0
                    verb_start_idx = 0
                    nom_start_idx = 0
                else:
                    word_start_idx = sent_len_cumsum[sent_id-1]
                    verb_start_idx = verbs_len_cumsum[sent_id-1]
                    nom_start_idx = nominals_len_cumsum[sent_id-1]
                word_end_idx = sent_len_cumsum[sent_id]
                verb_end_idx = verbs_len_cumsum[sent_id]
                nom_end_idx = nominals_len_cumsum[sent_id]
                sent_verb_info = {
                    'verbs': verb_info['verbs'][verb_start_idx:verb_end_idx], #verb_info['verbs'][verb_start_idx:verb_end_idx],
                    'words': verb_info['words'][word_start_idx:word_end_idx],
                    'saliency': list(),
                    'ranking': list(),
                    'total_pred': 0 if doc_id not in pred_doc_res else len(pred_doc_res[doc_id]['pred']),
                    'doc_id': verb_info['doc_id'],
                    'sent_idx': sent_id,
                }
                sent_nom_info = {
                    'nominals': nom_info['nominals'][nom_start_idx:nom_end_idx], #nom_info['nominals'][nominal_start_idx:nominal_end_idx],
                    'words': nom_info['words'][word_start_idx:word_end_idx],
                    'saliency': list(),
                    'ranking': list(),
                    'total_pred': 0 if doc_id not in pred_doc_res else len(pred_doc_res[doc_id]['pred']),
                    'doc_id': doc_id,
                    'sent_idx': sent_id,
                }
                if doc_id in pred_doc_res and verb_info['verb_index'][verb_end_idx-1] <= pred_doc_res[doc_id]['event_start_idx'][-1]:
                    pred_info = pred_doc_res[doc_id]
                    ranking = list(map(int, np.argsort(pred_info['pred'])[::-1]))
                    logits = pred_info['pred']
                    for verb_index in verb_info['verb_index'][verb_start_idx:verb_end_idx]:
                        cur_idx = pred_info['event_start_idx'].index(verb_index)
                        sent_verb_info['ranking'].append(ranking[cur_idx])
                        sent_verb_info['saliency'].append(logits[cur_idx])

                    for nomimal_index in nom_info['nominal_indices'][nom_start_idx:nom_end_idx]:
                        cur_idx = pred_info['event_start_idx'].index(nomimal_index[0])
                        sent_nom_info['ranking'].append(ranking[cur_idx])
                        sent_nom_info['saliency'].append(logits[cur_idx])
                sent_verb_list.append(sent_verb_info)
                sent_nom_list.append(sent_nom_info)
        assert len(sent_nom_list) == sum(1 for i in open(salient_nom_path))
        assert len(sent_verb_list) == sum(1 for i in open(salient_verb_path))
        os.makedirs(os.path.join(parent_folder, 'nyt'), exist_ok=True)
        fnom_out = open(salient_nom_path, 'w')
        fverb_out = open(salient_verb_path, 'w')
        for idx, (verb_line, nom_line, sent_verb, sent_nom) in enumerate(zip(open(verb_path), open(nom_path), sent_verb_list, sent_nom_list)):
            verb_info, nom_info = json.loads(verb_line), json.loads(nom_line)
            # assert verb_info['verbs'] == sent_verb['verbs']
            assert verb_info['words'] == sent_verb['words']
            # assert len(sent_verb['verbs']) == len(sent_verb['saliency'])
            fverb_out.write(json.dumps(sent_verb) + '\n')

            assert nom_info['words'] == sent_nom['words']
            # assert len(sent_nom['nominals']) == len(sent_nom['saliency'])
            fnom_out.write(json.dumps(sent_nom) + '\n')
        fverb_out.close()
        fnom_out.close()

    logger.info('step III: final top percentile events extraction')
    top_verb_path = os.path.join(parent_folder, 'nyt', f'verb_sense_srl_saliency_top{top_percentile}.txt')
    top_nom_path = os.path.join(parent_folder, 'nyt', f'nom_sense_srl_saliency_top{top_percentile}.txt')
    if not os.path.exists(top_verb_path) or overwrite:
        fverb_out = open(top_verb_path, 'w')
        # verb predicate: SpacyWordSplitter with VERB pos_; TODO: add index later to distinguish in long sentences
        num_verb_events, num_sents = 0, 0
        for line in open(salient_verb_path):
            instance = json.loads(line)
            saved_instance = {
                'verbs': list(),
                'words': instance['words'],
                'doc_id': instance['doc_id'],
                'sent_idx': instance['sent_idx']
            }
            for i, j in zip(instance['verbs'], instance['ranking']):
                if j <= int(top_percentile * instance['total_pred']):
                    saved_instance['verbs'].append(i)
                    num_verb_events += 1
            fverb_out.write(json.dumps(saved_instance)+'\n')
            num_sents += 1
        fverb_out.close()
        logger.info(f'num_verb_events: {num_verb_events}, num_sents: {num_sents}')

        fnom_out = open(top_nom_path, 'w')
        num_nom_events, num_sents = 0, 0
        for line in open(salient_nom_path):
            instance = json.loads(line)
            saved_instance = {
                'nominals': list(),
                'words': instance['words'],
                'doc_id': instance['doc_id'],
                'sent_idx': instance['sent_idx']
            }
            for i, j in zip(instance['nominals'], instance['ranking']):
                if j <= int(top_percentile * instance['total_pred']):
                    saved_instance['nominals'].append(i)
                    num_nom_events += 1
            fnom_out.write(json.dumps(saved_instance)+'\n')
            num_sents += 1
        fnom_out.close()
        logger.info(f'num_nom_events: {num_nom_events}, num_sents: {num_sents}')
    else:
        num_verb_events, num_sents = 0, 0
        for line in open(top_verb_path):
            instance = json.loads(line)
            num_verb_events += len(instance['verbs'])
            num_sents += 1
        logger.info(f'num_verb_events: {num_verb_events}, num_sents: {num_sents}')

        num_nom_events, num_sents = 0, 0
        for line in open(top_nom_path):
            instance = json.loads(line)
            num_nom_events += len(instance['nominals'])
            num_sents += 1
        logger.info(f'num_nom_events: {num_nom_events}, num_sents: {num_sents}')

    saved_verb_path = os.path.join(parent_folder, 'verb_sense_srl_saliency.txt')
    saved_nom_path = os.path.join(parent_folder, 'nom_sense_srl_saliency.txt')
    if not os.path.exists(saved_verb_path) or overwrite:
        preserve_events(corpus_path, salient_verb_path, salient_nom_path, saved_verb_path, saved_nom_path)
    return

def get_plain_embedding(corpus_path, res_folder, gpu_id, overwrite, source_list=['verb', 'nom']):
    parent_folder = os.path.join(res_folder, 'plain_embedding')
    os.makedirs(parent_folder, exist_ok=True)

    logger.info('step I: sbert embedding...')
    get_sent_sbert_repre(
        token_path=corpus_path,
        saved_folder=os.path.join(parent_folder, 'sentence_transformers'),
        gpu_id=gpu_id,
        batch_size=4,
        overwrite=overwrite)

    logger.info('step II: token-level embedding...')
    get_token_repre(
        token_path=corpus_path,
        saved_folder=os.path.join(parent_folder, 'bert_large_repre'),
        gpu_id=gpu_id,
        batch_size=10,
        overwrite=overwrite)

    logger.info('step II-I: event representation')
    extract_event_repre(
        source_folder=parent_folder,
        source_list=source_list,
        overwrite=overwrite,
        sense_srl_file_=os.path.join(res_folder, 'salient_event_detection', '{}_sense_srl_saliency.txt')
    )

    feature_saved_path = f'{parent_folder}/features_embedding.tsv'
    if not os.path.exists(feature_saved_path) or overwrite:
        combine_plain_embeddings(feature_saved_path, overwrite)
    return

def extract_external_embedding(res_folder, ref_events, device_id, overwrite):
    filename = './improve_embedding.yaml'
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config['trainer_params']['gpus'] = [device_id]

    cur_config = deepcopy(config)

    saved_dir = f'{res_folder}/improved_embedding'
    os.makedirs(saved_dir, exist_ok=True)
    feature_path = os.path.join(saved_dir, 'features_embedding.tsv')
    if not os.path.exists(feature_path) or overwrite:
        run_embedding(cur_config, f'./models/ae/{ref_events}', saved_dir)
    return

def hierarchical_clustering(res_folder, algorithm, seed, overwrite):
    assert algorithm == 'linkage_ward'

    logger.info('step I: extract features')
    perf_path = os.path.join(res_folder, 'hierarchical_clustering', f'perf.json')
    tree_path = perf_path.replace('perf.json', 'tree.tsv')
    logger.info(f'perf_path: {perf_path}')
    os.makedirs(os.path.dirname(perf_path), exist_ok=True)

    feature_saved_path = os.path.join(res_folder, 'improved_embedding', 'features_embedding.tsv')
    assert os.path.exists(feature_saved_path)

    logger.info('step II: perform hierarchical clustering')
    if not os.path.exists(tree_path) or overwrite:
        generate_linkage_ward_tree(
            pred_path=tree_path,
            data_path=feature_saved_path,
            random_seed=seed,
            overwrite=overwrite)
    return

def name_generation(corpus_path, res_folder, method, generator_model, demo_dataset, seed, overwrite):
    assert method == 'generator'
    param_grid = {
        'k': 64,
        'source': 'predicate',
        'use_calibration': False,
        'son_flag': False,
        'sampling': False,
        'demo_dataset': demo_dataset,
        'model': generator_model,
        'template': {
             'input': ' sentence: ',
             'predicate': ' predicate: ',
             'output': ' event type: ',
        },
         'overwrite': overwrite,
         'spacy_model': 'en_core_web_lg',
          'seed': seed}

    tree_path = f'{res_folder}/hierarchical_clustering/tree.tsv'
    saved_folder = f'{res_folder}/name'
    os.makedirs(saved_folder, exist_ok=True)


    features_embedding = f'{res_folder}/improved_embedding/features_embedding.tsv'
    num_samples = sum(1 for i in open(features_embedding))
    tree = convert_tree_file_to_nx_tree(tree_path, tree_path.replace('tsv', 'gpickle'),
                                        num_samples,
                                        overwrite,
                                        save_flag=True)
    leaves = [i for i in tree.nodes if tree.out_degree[i] == 0]
    assert np.array_equal(np.sort(leaves), np.arange(num_samples))
    root_candidates = [i for i, j in tree.in_degree if j == 0]
    assert len(root_candidates) == 1
    root_node = root_candidates[0]
    p = nx.shortest_path(tree)

    root_dists = nx.shortest_path_length(tree, root_node)

    feature_info = pickle.load(open(f'{res_folder}/plain_embedding/arg1_emb_arg2_emb_predicate_emb_sense_example_emb_sense_explanation_emb_sent_sbert_emb_average/samples.pkl', 'rb'))
    name_path = os.path.join(saved_folder, 'name.json')
    sample_dict = {
        'nom': pickle.load(open(f'{res_folder}/plain_embedding/nom_arg_sense_emb.pkl', 'rb')),
        'verb': pickle.load(open(f'{res_folder}/plain_embedding/verb_arg_sense_emb.pkl', 'rb')),
    }
    if not os.path.exists(name_path) or overwrite:
        res = generative(saved_folder=saved_folder, feature_info=feature_info, sample_dict=sample_dict,
                         tree=tree, root_dists=root_dists, p=p, root_node=root_node,
                         param_grid=param_grid)
        fout = open(name_path, 'w')
        for ins in res:
            fout.write(json.dumps(ins)+'\n')
        fout.close()

    combined_name_path = os.path.join(saved_folder, 'event_type_names.json')
    if not os.path.exists(combined_name_path) or overwrite:
        combine_event_info_with_name(corpus_path, saved_folder)

    config_path = os.path.join(saved_folder, 'config.json')
    if not os.path.exists(config_path) or overwrite:
        with open(os.path.join(saved_folder, 'config.json'), 'w') as f:
            json.dump(param_grid, f, indent=4)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', '-c', type=str, default='./datasets/allsides_lgbt.txt', help='path of open corpus')
    parser.add_argument('--device_id', '-id', type=int, default=0, help='either -1 for using cpu or non-negative values indicating gpu id')
    parser.add_argument('--res_folder', '-f', type=str, default='')
    parser.add_argument('--overwrite', '-o', action='store_true', help='whether to overwrite results')
    parser.add_argument('--seed', '-s', type=int, default=0, help='random seed')

    # parameters for saliency
    parser.add_argument('--top_percentile', '-t', type=float, default=0.1, help='top events considered salient')
    # parameters for embedding
    parser.add_argument('--ref_events', '-re', type=str, choices=['ace2005', 'maven', 'rams'], default='maven')
    # parameters for name generation
    parser.add_argument('--generator_model', '-gm', type=str, default='gpt-j-6b', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'gpt-j-6b'])
    parser.add_argument('--demo_dataset', '-demo', type=str, default='maven')


    args = parser.parse_args()

    if args.res_folder == '':
        args.res_folder = os.path.join('./results', args.corpus_path.split('/')[-1].rstrip('.txt'))
    os.makedirs(args.res_folder, exist_ok=True)

    logger.info(f'result folder: {args.res_folder}...')
    logger.info(f'result overwrite: {args.overwrite}...')

    logger.info('step I: event extraction...')
    event_extraction(corpus_path=args.corpus_path, res_folder=args.res_folder, gpu_id=args.device_id, overwrite=args.overwrite)
    print('*'*60)

    logger.info('step II: salient_event_detection...')
    salient_event_detection(corpus_path=args.corpus_path, res_folder=args.res_folder,
                            top_percentile=args.top_percentile, gpu_id=args.device_id,
                            seed=args.seed, overwrite=args.overwrite)
    print('*'*60)

    logger.info('step III: get plain embedding...')
    get_plain_embedding(corpus_path=args.corpus_path, res_folder=args.res_folder, gpu_id=args.device_id,
                  overwrite=args.overwrite)
    print('*'*60)

    logger.info('step IV: get improved embedding...')
    extract_external_embedding(args.res_folder, args.ref_events, args.device_id, args.overwrite)
    print('*'*60)

    logger.info('step V: hierarchical clustering...')
    hierarchical_clustering(res_folder=args.res_folder, algorithm='linkage_ward', seed=args.seed, overwrite=args.overwrite)
    print('*'*60)

    logger.info('step VI: type name generation...')
    name_generation(corpus_path=args.corpus_path, res_folder=args.res_folder, method='generator', generator_model=args.generator_model,
                    demo_dataset=args.demo_dataset, seed=args.seed, overwrite=args.overwrite)


