# Created by ceoeventontology at 2023/2/9
import os
import numpy as np
import json
import pickle
from scipy.special import softmax
from tqdm import tqdm
from copy import deepcopy
import psutil
from event_data_hierarchies import son_parent_type
import networkx as nx
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from open_utilities import nlp, _set_random_seed
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM
import logging
logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger.setLevel(logging.DEBUG)

train_file_name = {
    'ace2005': 'train.oneie.json',
    'maven': 'train.jsonl',
    'rams': 'train.jsonlines',
}

def _eval_node_id(node_id):
    try:
        _node_id = eval(node_id)
    except:
        _node_id = node_id
    return _node_id

def convert_tree_file_to_nx_tree(tree_path, nx_tree_path, n, overwrite, save_flag=False):
    if os.path.exists(nx_tree_path) and not overwrite:
        logger.info(f'directly load tree saved at {nx_tree_path}')
        tree = nx.read_gpickle(nx_tree_path)
        return tree
    tree = nx.DiGraph()

    with open(tree_path, 'r') as f:
        tree_info = f.readlines()

    internal_id_map = dict()
    count = 0
    child_parent_dict = dict()
    root_node = None
    for line in tree_info:
        child_node, parent_node, child_label = line.rstrip('\n').split('\t')
        child_node, parent_node, child_label = _eval_node_id(child_node), _eval_node_id(parent_node), _eval_node_id(child_label)
        if parent_node is None:
            root_node = child_node
            continue
        if child_label is None:
            # this child is internal node
            internal_id_map[child_node] = count
            count += 1
        child_parent_dict[child_node] = parent_node
    assert root_node is not None
    internal_id_map[root_node] = count

    for child_node, parent_node in child_parent_dict.items():
        if child_node in internal_id_map:
            actual_child_node = internal_id_map[child_node] + n
        else:
            actual_child_node = child_node
        # parent node could not be sample node
        actual_parent_node = internal_id_map[parent_node] + n
        tree.add_edge(actual_parent_node, actual_child_node)

    if save_flag:
        nx.write_gpickle(tree, nx_tree_path)

    logger.info(f'tree with {len(tree.nodes())} nodes saved at {nx_tree_path}')
    return tree

def label_split(label, dataset):
    if dataset == 'ace2005':
        res = label.split(':')
    elif dataset == 'rams':
        res = label.split('.')
        while res[-1] == 'n/a':
            res = res[:-1]
    elif dataset == 'maven':
        res = [label]
        parent = son_parent_type[dataset][label][0]
        res.append(parent)
        while parent in son_parent_type[dataset]:
            parent = son_parent_type[dataset][parent][0]
            res.append(parent)
        res = res[::-1][1:]#[:gold_hierarchies[dataset]]
    return res

def create_demonstrations(dataset):
    info_list = list()
    for line in open(f'./resources/event_datasets/{dataset}/{train_file_name[dataset]}', 'r'):
        info_list.append(json.loads(line))
    if dataset == 'maven':
        new_info_list = list()
        for sample in tqdm(info_list):
            sent_dict, log_dict = dict(), dict()
            for events in sample['events']:
                for mention in events['mention']:
                    if mention['sent_id'] not in sent_dict:
                        sent_dict[mention['sent_id']] = {
                            'sentence': sample['content'][mention['sent_id']]['sentence'],
                            'event_mentions': [
                                {
                                    'event_type': events['type'],
                                    'trigger': {'text': mention['trigger_word']},
                                }
                            ]
                        }
                        log_dict[mention['sent_id']] = ['_'.join([events['type'], mention['trigger_word']])]
                    else:
                        if '_'.join([events['type'], mention['trigger_word']]) not in log_dict[mention['sent_id']]:
                            sent_dict[mention['sent_id']]['event_mentions'].append({
                                'event_type': events['type'],
                                'trigger': {'text': mention['trigger_word']},
                            })
                            log_dict[mention['sent_id']].append('_'.join([events['type'], mention['trigger_word']]))

            new_info_list += list(sent_dict.values())
        info_list = deepcopy(new_info_list)
    elif dataset == 'rams':
        new_info_list = list()
        for sample in tqdm(info_list):
            sent_len = list(map(len, sample['sentences']))
            sent_len_sum = np.cumsum(sent_len)

            sent_dict = dict()
            for evt_triggers in sample['evt_triggers']:
                sent_id = sorted(np.argwhere(sent_len_sum > evt_triggers[0]).reshape([-1]))[0]
                for type_list in evt_triggers[2]:
                    if sent_id not in sent_dict:
                        sent_dict[sent_id] = {
                            'sentence': ' '.join(sample['sentences'][sent_id]),
                            'event_mentions': [
                                {
                                    'event_type': type_list[0],
                                    'trigger': {
                                        'text': sample['sentences'][sent_id][evt_triggers[0]-sent_len_sum[sent_id-1]] if sent_id > 0 else sample['sentences'][sent_id][evt_triggers[0]]
                                    }
                                }
                            ]
                        }
                    else:
                        sent_dict[sent_id]['event_mentions'].append(
                            {
                                'event_type': type_list[0],
                                'trigger':
                                    {
                                        'text': sample['sentences'][sent_id][
                                    evt_triggers[0] - sent_len_sum[sent_id - 1]] if sent_id > 0 else
                                sample['sentences'][sent_id][evt_triggers[0]]}
                            }
                        )
            new_info_list += list(sent_dict.values())
        info_list = deepcopy(new_info_list)

    fout_sentence = open(os.path.join(f'./resources/in_context_learning', f'{dataset}.demo_sentence.json'), 'w')
    fout_predicate = open(os.path.join(f'./resources/in_context_learning', f'{dataset}.demo_predicate.json'), 'w')

    lines = 0
    for sample in info_list:
        for event_mention in sample['event_mentions']:
            cur_instance = {
                'input': sample['sentence'],
                'output': label_split(event_mention['event_type'], dataset)[-1],
            }
            fout_sentence.write(json.dumps(cur_instance)+'\n')
            # cur_instance['input'] += f' The predicate is {event_mention["trigger"]["text"]}.'
            cur_instance['predicate'] = event_mention["trigger"]["text"]
            fout_predicate.write(json.dumps(cur_instance)+'\n')
            lines += 1
    fout_sentence.close()
    fout_predicate.close()
    logger.info(f'{lines} lines saved at {os.path.join(f"./resources/in_context_learning", f"{dataset}.demo_sentence/predicate.json")}')

    return

def create_prompt(sample_dict, saved_dir):
    for predicate_type in ['verb', 'nom']:
        fout_sentence = open(os.path.join(saved_dir, f'{predicate_type}_shot_sentence.json'), 'w')
        fout_predicate = open(os.path.join(saved_dir, f'{predicate_type}_shot_predicate.json'), 'w')

        lines = 0
        source_data = sample_dict[predicate_type]
        for sample in source_data:
            cur_instance = {
                'input': ' '.join(sample['words']),
            }
            fout_sentence.write(json.dumps(cur_instance)+'\n')
            cur_instance['predicate'] = sample["predicate_lemma"][0]
            fout_predicate.write(json.dumps(cur_instance)+'\n')
            lines += 1
        fout_sentence.close()
        fout_predicate.close()
        logger.info(f'{lines} lines saved at {os.path.join(saved_dir, f"train.{predicate_type}_shot_sentence/predicate.json")}')
    return

class few_shot_generation:
    def __init__(self, model, use_demonstrations, k, max_length, template, max_length_per_example=256, method='direct'):
        self.method = method
        self.add_newlines = False
        if model.startswith('gpt2'):
            self.model = AutoModelForCausalLM.from_pretrained(model)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        elif 'gpt-j' in model:
            num_gpus = torch.cuda.device_count()
            max_memory = {i: "20GB" for i in range(num_gpus)}
            max_memory["cpu"] = psutil.virtual_memory().available
            self.model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        if torch.cuda.device_count() > 0:
            self.model.cuda()
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')
        self.model.eval()
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length_per_example = max_length_per_example
        if self.use_demonstrations:
            max_length = min(max_length * self.k, 1024)
        self.max_length = max_length

        self.template = template
        self.output_length = len(self.tokenizer(template['output'])['input_ids'])
        if self.template['output'].endswith(' '):
            self.output_length -= 1

    def load_data(self, data_paths, is_null=False, train_flag=False, test_size=None):
        if type(data_paths) == str:
            data_paths = [data_paths]
        data = list()
        for data_path in data_paths:
            with open(data_path, "r") as f:
                for line in f:
                    dp = json.loads(line)
                    if is_null:
                        dp["input"] = "N/A"
                    data.append(dp)
        if train_flag and self.use_demonstrations and test_size is not None:
            selected_idx = np.random.choice(range(len(data)), test_size*self.k, replace=True)
            # selected_idx = np.random.choice(range(len(data)), self.k, replace=True)
            data = [data[i] for i in selected_idx]
        logger.info(f'load {len(data)} samples from {data_paths}...')
        return data

    def prepro_sentence_pair_single(self, ids1, ids2, max_length,
                                    allow_truncation=False):
        # assert len(ids2) == 1
        if allow_truncation and len(ids1) + len(ids2) > max_length:
            ids1 = ids1[len(ids1) + len(ids2) - max_length:]  # len = max_length-len(ids2)
            assert len(ids1) + len(ids2) == max_length

        n_mask = max_length - len(ids1) - len(ids2)
        assert n_mask >= 0, (max_length, len(ids1), len(ids2))
        input_ids = ids1 + ids2 + [0 for _ in range(n_mask)]
        attention_mask = [1 for _ in ids1 + ids2] + [0 for _ in range(n_mask)]
        token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
        return input_ids, attention_mask, token_type_ids

    def _prepro_each_datapoint(self, dp_, is_first=True, is_training=False, for_demonstrations=False,
                               add_newlines=True):
        dp = dp_.copy()
        assert not add_newlines
        if not is_first:
            if self.method=="direct":
                dp["input"] = self.template['input'] + dp["input"]
            elif self.method=="channel":
                dp["output"] = " " + dp["output"]
                if "options" in dp:
                    dp["options"] = [" "+opt for opt in dp["options"]]
            else:
                raise NotImplementedError()
        if self.method=="direct":
            if 'output' in dp:
                dp["output"] = self.template['output'] + dp["output"]
            if "options" in dp:
                dp["options"] = [" " + opt for opt in dp["options"]]
        elif self.method=="channel":
            dp["input"] = " " + dp["input"]
        else:
            raise NotImplementedError()

        if 'predicate' in dp:
            if dp_['input'] =='N/A':
                dp['predicate'] = 'N/A'
            dp['input'] += self.template['predicate'] + dp['predicate']

        input_tokens = self.tokenizer(dp["input"])["input_ids"]

        if is_training or for_demonstrations:
            output_tokens = self.tokenizer(dp["output"])["input_ids"]

            if len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, \
                (dp.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)

            if self.method=="direct":
                return input_tokens, output_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()

        else:
            option_length = len(self.tokenizer('happy')["input_ids"])
            truncation = 0
            if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
                input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]
                truncation = 1

            # input_tokens = [input_tokens for _ in option_tokens]

            output_tokens = self.tokenizer(self.template['output'])["input_ids"]

            if self.method=="direct":
                return input_tokens, output_tokens, truncation

    def tensorize(self, _train_data, _test_data):
        add_newlines = False
        train_data, test_data = [], []
        if self.use_demonstrations:
            for dp in _train_data:
                assert type(dp)==dict, ("Each example should be a dictionary", dp)
                assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                train_data.append(dp.copy())
        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp, \
                ("Test example should contain input and options in a list format", dp)
            test_data.append(dp.copy())

        input_ids, attention_mask, token_type_ids = [], [], []

        if self.use_demonstrations:
            assert len(train_data) >= self.k
        truncation = 0
        for dp_idx, dp in tqdm(enumerate(test_data), total=len(test_data)):
            inputs, outputs, truncation_ = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)
            truncation += truncation_

            # for inputs_ in inputs:
            if self.use_demonstrations:
                demonstrations = []
                for i, dp in enumerate(train_data[dp_idx*self.k:(dp_idx+1)*self.k]):
                    input_, output_ = self._prepro_each_datapoint(
                        dp, is_first=i == 0, for_demonstrations=True,
                        add_newlines=add_newlines)
                    demonstrations += input_ + output_
                inputs = demonstrations + inputs

            encoded = self.prepro_sentence_pair_single(
                inputs, outputs, self.max_length,
                allow_truncation=self.use_demonstrations)

            input_ids.append(encoded[0])
            attention_mask.append(encoded[1])
            token_type_ids.append(encoded[2])

        tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        return tensorized_inputs, truncation/len(test_data)

    def print_tensorized_example(self, tensorized_inputs, return_string=False):
        idx = np.random.choice(range(len(tensorized_inputs['input_ids'])), 1, replace=False)[0]
        text = f"Checking {idx}-th example..."
        input_ids = tensorized_inputs["input_ids"][idx]
        token_type_ids = tensorized_inputs["token_type_ids"][idx]
        if type(input_ids)!=list:
            input_ids = input_ids.numpy().tolist()
        if type(token_type_ids)!=list:
            token_type_ids = token_type_ids.numpy().tolist()

        text += "\nInput:\n"
        if 1 in token_type_ids:
            text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
            text += "\nOutput:\n"
            text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id == 1])
        else:
            text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])

        if return_string:
            return text

        logger.info(text)
        return

    def get_dataloader(self, tensorized_inputs, batch_size, is_training):
        inputs = tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        logger.info(shape)
        for v in inputs.values():
            assert v.shape == shape
        if "labels" in inputs:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["labels"])
        else:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        if is_training:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def do_inference(self, tensorized_inputs, batch_size=1, verbose=False):
        dataloader = self.get_dataloader(tensorized_inputs, batch_size, is_training=False)
        if verbose:
            dataloader = tqdm(dataloader)
        probs = []
        for batch in tqdm(dataloader, total=int(len(tensorized_inputs['input_ids'])//batch_size)):
            input_ids = batch[0]
            attention_mask = batch[1]
            token_type_ids = batch[2]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
            with torch.no_grad():
                cur_probs = self.run_model(input_ids, attention_mask, token_type_ids)
            probs += cur_probs.cpu().detach().numpy().tolist()
        probs = np.stack(probs, axis=0)
        return probs


    def do_predict(self, prob_list, sampling=False, top_k=1):
        results = []
        for probs in prob_list:
            if sampling:
                predictions = np.random.choice(np.arange(len(probs)), size=top_k, replace=False, p=probs)
                values = probs[predictions]
            else:
                values, predictions = torch.from_numpy(probs).to(self.device).topk(top_k)
            result = []
            for v, p in zip(values.tolist(), predictions.tolist()):
                result.append(
                    {
                        "score": v,
                        "token_str": self.tokenizer.decode(p).strip(),
                    }
                )
            results += [result]
        if len(results) == 1:
            return results[0]
        return results

    def run_model(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits_list = list()
        for logit, mask in zip(outputs.logits, token_type_ids):
            label_idx = torch.sort(torch.where(mask == 1)[0])[0]
            assert len(label_idx) in [self.output_length, self.output_length+1]
            logits_list.append(logit[label_idx[self.output_length-1]])
        logits_list = torch.stack(logits_list, dim=0)
        probs = F.softmax(logits_list, dim=-1)
        return probs

class few_shot_decoding:
    def __init__(self, model):
        if model.startswith('gpt2'):
            model_name = model
        elif 'gpt-j' in model:
            model_name = 'EleutherAI/gpt-j-6B'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if torch.cuda.device_count() > 0:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')

    def do_predict(self, prob_list, sampling=False, top_k=1):
        results = []
        for probs in prob_list:
            if sampling:
                predictions = np.random.choice(np.arange(len(probs)), size=top_k, replace=False, p=probs)
                values = probs[predictions]
            else:
                values, predictions = torch.from_numpy(probs).to(self.device).topk(top_k)
            result = []
            for v, p in zip(values.tolist(), predictions.tolist()):
                result.append(
                    {
                        "score": v,
                        "token_str": self.tokenizer.decode(p).strip(),
                    }
                )
            results += [result]
        if len(results) == 1:
            return results[0]
        return results

def generative(saved_folder, feature_info, sample_dict, tree, root_dists, p, root_node, param_grid):
    res_folder = os.path.dirname(saved_folder)

    sampling, son_flag, demo_dataset = param_grid['sampling'], param_grid['son_flag'], param_grid['demo_dataset']
    model, k, source, use_calibration, template = param_grid['model'], param_grid['k'], param_grid['source'], param_grid['use_calibration'], param_grid['template']
    overwrite = param_grid['overwrite']
    assert use_calibration == False
    assert source == 'predicate'

    seed = param_grid['seed']
    _set_random_seed(seed)

    logger.info('step I: demonstrations')
    train_path = f'./resources/in_context_learning/{demo_dataset}.demo_{source}.json'
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    if not os.path.exists(train_path):
        create_demonstrations(demo_dataset)

    logger.info('step II: test instances')
    verb_test_path = f'{res_folder}/salient_event_detection/in_context_learning/verb_shot_{source}.json'
    nom_test_path = f'{res_folder}/salient_event_detection/in_context_learning/nom_shot_{source}.json'
    os.makedirs(os.path.dirname(verb_test_path), exist_ok=True)
    if not os.path.exists(verb_test_path) or os.path.exists(nom_test_path):
        create_prompt(sample_dict=sample_dict, saved_dir=os.path.dirname(verb_test_path))

    logger.info('step III: inference')
    verb_prob_path = os.path.join(saved_folder, 'verb_probs.pkl')
    nom_prob_path = os.path.join(saved_folder, 'nom_probs.pkl')
    if not os.path.exists(verb_prob_path) or not os.path.exists(nom_prob_path) or overwrite:
        generator = few_shot_generation(
            model=model,
            use_demonstrations=True,
            max_length=1024,
            k=k,
            template=template
        )
        _train_data = generator.load_data(data_paths=train_path, is_null=False, train_flag=True)
        prob = dict()
        for predicate_type in ['verb', 'nom']:
            _test_data = generator.load_data(data_paths=locals()[f'{predicate_type}_test_path'], is_null=False)
            selected_idx = np.random.choice(range(len(_train_data)), len(_test_data)*k, replace=True)
            local_train_data = [_train_data[i] for i in selected_idx]
            tensorized_inputs, truncation_rate = generator.tensorize(local_train_data, _test_data)
            generator.print_tensorized_example(tensorized_inputs)
            generator.print_tensorized_example(tensorized_inputs)
            actual_probs = generator.do_inference(tensorized_inputs, batch_size=1)
            prob[predicate_type] = actual_probs
            with open(os.path.join(saved_folder, f'{predicate_type}_probs.pkl'), 'wb') as f:
                pickle.dump(actual_probs, f, protocol=4)

            with open(os.path.join(saved_folder, f'{predicate_type}_truncation_calibrated.json'), 'w') as f:
                json.dump({'truncation_rate': truncation_rate}, f)
    else:
        prob = {
            'verb': pickle.load(open(os.path.join(saved_folder, 'verb_probs.pkl'), 'rb')),
            'nom': pickle.load(open(os.path.join(saved_folder, 'nom_probs.pkl'), 'rb')),
        }
    assert len(feature_info['X']) == len(prob['verb']) + len(prob['nom']) == len(sample_dict['verb']) + len(sample_dict['nom'])

    logger.info('step IV: node name generation')
    node_prob_dict, node_prob_full_dict, node_count_full_dict = dict(), dict(), dict()
    for ele in sorted(root_dists.items(), key=lambda x: -x[1]):
        node_id = ele[0]
        if node_id < len(feature_info['X']):
            source_name, source_idx = feature_info['source_name'][node_id], feature_info['source_idx'][node_id]
            node_prob_dict[node_id] = prob[source_name][source_idx]
            node_prob_full_dict[node_id] = node_prob_dict[node_id]
            node_count_full_dict[node_id] = 1
        else:
            node_prob_full_dict[node_id] = None
            node_count_full_dict[node_id] = 0
            node_prob_dict[node_id] = list()
            for idx, son_id in enumerate(tree.adj[node_id]):
                # tmp.append(node_prob_dict[son_id])
                if node_prob_full_dict[node_id] is None:
                    node_prob_full_dict[node_id] = node_prob_full_dict[son_id] * node_count_full_dict[son_id]
                else:
                    node_prob_full_dict[node_id] += node_prob_full_dict[son_id] * node_count_full_dict[son_id]
                node_count_full_dict[node_id] += node_count_full_dict[son_id]
                # node_prob_full_dict[node_id].append(node_prob_full_dict[son_id])
                node_prob_dict[node_id].append(node_prob_dict[son_id])
            node_prob_dict[node_id] = softmax(np.mean(np.stack(node_prob_dict[node_id], axis=0), axis=0))
            # node_prob_full_dict[node_id] = np.concatenate(node_prob_full_dict[node_id], axis=0)
            node_prob_full_dict[node_id] /= node_count_full_dict[node_id]

    res = list()
    decoder = few_shot_decoding(model)
    for i in tqdm(range(len(feature_info['X']))):
        source_name, source_idx = feature_info['source_name'][i], feature_info['source_idx'][i]
        sample = sample_dict[source_name][source_idx]

        cur_shortest_path = p[root_node][i]
        if 'predicate_index' in sample:
            predicate_index = sample['predicate_index']
            predicate_text = ' '.join([sample['words'][i] for i in predicate_index])
        else:
            predicate_index = [sample['verb_index']]
            predicate_text = sample['words'][sample['verb_index']]
        res.append({
            'line_idx': sample['line_idx'],
            'predicate_lemma': sample['predicate_lemma'],
            'predicate_index': predicate_index,
            'predicate_text': predicate_text,
            'pred': list(),
            'words': sample['words']
        })
        for node_id in cur_shortest_path[1:-1]:
            if son_flag:
                node_prob = np.array([node_prob_dict[node_id]])
            else:
                # node_prob = np.array([softmax(np.mean(node_prob_full_dict[node_id], axis=0))])
                node_prob = np.array([softmax(node_prob_full_dict[node_id])])
            predictions = decoder.do_predict(node_prob, sampling=sampling, top_k=100) # first instance
            pred_lemma, counts = None, 0
            while pred_lemma is None and counts < len(predictions):
                pred = predictions[counts]['token_str']
                try:
                    doc = nlp([pred])
                    if doc[0].pos_ in ['NOUN', 'VERB']:
                        pred_lemma = doc[0].lemma_
                    else:
                        pred_lemma = None
                        counts += 1
                except:
                    pred_lemma = None
                    counts += 1
            if pred_lemma is None:
                # have to
                pred_lemma = nlp(predictions[0]['token_str'])[0].lemma_
            res[i]['pred'].append(pred_lemma)
        concise_pred = list()
        for j in range(len(res[i]['pred'])):
            if res[i]['pred'][j] not in concise_pred:
                concise_pred.append(res[i]['pred'][j])
        res[i]['pred'] = concise_pred
    return res

def combine_event_info_with_name(corpus_path, saved_folder):
    saved_info_dict = dict()
    for line in open(os.path.join(saved_folder, 'name.json')):
        res = json.loads(line)
        if res['line_idx'] not in saved_info_dict:
            saved_info_dict[res['line_idx']] = {
                'tokens': res['words'],
                'predictions': [
                    {
                        'predicate_lemma': res['predicate_lemma'],
                        'predicate_index': res['predicate_index'],
                        'predicate_text': res['predicate_text'],
                        'type_names': ':'.join(res['pred']),
                    }
                ]
            }
        else:
            assert saved_info_dict[res['line_idx']]['tokens'] == res['words']
            saved_info_dict[res['line_idx']]['predictions'].append({
                'predicate_lemma': res['predicate_lemma'],
                'predicate_index': res['predicate_index'],
                'predicate_text': res['predicate_text'],
                'type_names': ':'.join(res['pred']),
            })

    fout = open(os.path.join(saved_folder, 'event_type_names.json'), 'w')
    sentence_event_counts = list()
    for line_idx, line in enumerate(open(corpus_path)):
        source_info = json.loads(line)
        fout.write(json.dumps({
            'doc_id': source_info['doc_id'],
            'sent_id': source_info['sent_id'],
            'predictions': [] if line_idx not in saved_info_dict else saved_info_dict[line_idx]['predictions'],
            'tokens': source_info['tokens'],
            'sentence': source_info['sentence'],
        })+'\n')
        sentence_event_counts.append(line_idx in saved_info_dict)
    fout.close()
    logger.info('on average, %.2f%% of sentences have predicted events...'%(np.mean(sentence_event_counts)*100))
    # on average, 39% of sentences have predicted events...
    return





