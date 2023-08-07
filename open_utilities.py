# Created by ceoeventontology at 2023/2/7
"""
uitlity functions for open_corpus.py
"""
import os
import random
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
import logging
logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger.setLevel(logging.DEBUG)
import pickle
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from typing import Union, List
nlp = spacy.load('en_core_web_lg')
class _PretokenizedTokenizer:
    """Custom tokenizer to be used in spaCy when the text is already pretokenized."""
    """
    https://github.com/explosion/spaCy/issues/5399
    """
    def __init__(self, vocab: Vocab):
        """Initialize tokenizer with a given vocab
        :param vocab: an existing vocabulary (see https://spacy.io/api/vocab)
        """
        self.vocab = vocab

    def __call__(self, inp: Union[List[str], str]) -> Doc:
        """Call the tokenizer on input `inp`.
        :param inp: either a string to be split on whitespace, or a list of tokens
        :return: the created Doc object
        """
        if isinstance(inp, str):
            words = inp.split()
            spaces = [True] * (len(words) - 1) + ([True] if inp[-1].isspace() else [False])
            return Doc(self.vocab, words=words, spaces=spaces)
        elif isinstance(inp, list):
            return Doc(self.vocab, words=inp)
        else:
            raise ValueError("Unexpected input format. Expected string to be split on whitespace, or list of tokens.")
nlp.tokenizer = _PretokenizedTokenizer(nlp.vocab)
inference_include_columns_dict = {
    'bert-base-uncased': ['input_ids', 'token_type_ids', 'attention_mask'],# 'labels'],
    'bert-large-uncased-whole-word-masking': ['input_ids', 'token_type_ids', 'attention_mask'],  # 'labels'],
    'roberta-large': ['input_ids', 'attention_mask'],
    'sentence-transformers/all-roberta-large-v1': ['input_ids', 'attention_mask'],
    'deberta-v3-small': ['input_ids', 'token_type_ids', 'attention_mask'],# 'labels'],
    'deberta-base-mnli': ['input_ids', 'token_type_ids', 'attention_mask'],# 'labels'],
    'deberta-base': ['input_ids', 'token_type_ids', 'attention_mask'],# 'labels'],
    'deberta-large': ['input_ids', 'token_type_ids', 'attention_mask'],# 'labels'],
}

sense_path = {
    'verb': './resources/senses/word_pos_senses_explanation_examples_emb.pkl',
    'nom': './resources/senses/word_senses_explanation_examples_emb.pkl',
}

def _set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return

def convert_tokens(corpus_path, res_folder):
    with open(corpus_path, 'r') as f:
        sentences = f.readlines()
    saved_path = os.path.join(res_folder, 'tokens_srl.txt')
    fout = open(saved_path, 'w')
    for sentence in sentences:
        fout.write(json.dumps({'tokens': json.loads(sentence)['tokens']})+'\n')
    fout.close()
    return saved_path

def _shift_indices_for_empty_strings(words, indices):
    shiftleft = 0
    new_indices = []
    new_words = []
    for idx, word in enumerate(words):
        if word=="" or word.isspace():
            shiftleft += 1
        else:
            if idx in indices:
                new_indices.append(idx-shiftleft)
            new_words.append(word)
    return new_words, new_indices

def convert_id_to_srl_input(filename, outfile, tokens_output=True):
    with open(filename, 'r') as id_file, open(outfile, 'w') as outfile:
        for entry in id_file:
            data = json.loads(entry)
            indices = [idx for idx in range(len(data["nominals"])) if data["nominals"][idx]==1]
            if tokens_output:
                outfile.write(json.dumps({"tokens": data["words"], "indices": indices})+"\n")
            else:
                new_words, new_indices = _shift_indices_for_empty_strings(data["words"], indices)
                outfile.write(json.dumps({"sentence": " ".join(new_words), "indices": new_indices})+"\n")
    return

def salient_preparation(corpus_path, nom_path, verb_path, nom_doc_path, verb_doc_path):
    num_tokens = sum(1 for i in open(corpus_path))
    num_noms = sum(1 for i in open(nom_path))
    num_verbs = sum(1 for i in open(verb_path))
    assert num_tokens == num_noms == num_verbs

    noms_dict, verbs_dict = dict(), dict()
    for tokens_line, nom_line, verb_line in tqdm(zip(open(corpus_path), open(nom_path), open(verb_path)), total=num_tokens):
        tokens_info, nom_info, verb_info = json.loads(tokens_line), json.loads(nom_line), json.loads(verb_line)
        assert tokens_info['tokens'] == nom_info['words'] == verb_info['words']
        sent_id = tokens_info['sent_id']
        if len(nom_info['nominals']) > 0:
            noms_dict[sent_id] = {
                'nominals': nom_info['nominals'],
                'nominals_len': [len(nom_info['nominals'])],
                'nominal_indices': [i['nominal_indices'] for i in nom_info['nominals']],
            }
        if len(verb_info['verbs']) > 0:
            verbs_dict[sent_id] = {
                'verbs': verb_info['verbs'],
                'verbs_len': [len(verb_info['verbs'])],
                'verb_index': [i['verb_index'] for i in verb_info['verbs']]
            }

    verb_sample_list, nom_sample_list = list(), list()
    cur_nom_instance, cur_verb_instance = None, None
    prior_doc_id, prior_sent_num = None, None
    doc_id_list = list()
    for line in open(corpus_path):
        raw_info = json.loads(line)
        doc_id, sent_id = raw_info['doc_id'], raw_info['sent_id']
        sent_num = int(sent_id.split('-')[-1])
        doc_id_list.append(doc_id)
        if doc_id != prior_doc_id:
            if cur_nom_instance is not None:
                nom_sample_list.append(cur_nom_instance)
            if cur_verb_instance is not None:
                verb_sample_list.append(cur_verb_instance)
            cur_nom_instance = {
                'words': deepcopy(raw_info['tokens']),
                'sent_len': [len(raw_info['tokens'])],
                'doc_id': doc_id,
            }
            cur_verb_instance = {
                'words': deepcopy(raw_info['tokens']),
                'sent_len': [len(raw_info['tokens'])],
                'doc_id': doc_id,
            }
            if sent_id not in noms_dict:
                cur_nom_instance['nominals'] = []
                cur_nom_instance['nominals_len'] = [0]
                cur_nom_instance['nominal_indices'] = []
            else:
                cur_nom_instance.update(noms_dict[sent_id])
            if sent_id not in verbs_dict:
                cur_verb_instance['verbs'] = []
                cur_verb_instance['verbs_len'] = [0]
                cur_verb_instance['verb_index'] = []
            else:
                cur_verb_instance.update(verbs_dict[sent_id])
        else:
            assert sent_num == prior_sent_num + 1
            prior_nom_word_counts = len(cur_nom_instance['words'])
            cur_nom_instance['words'] += raw_info['tokens']
            cur_nom_instance['sent_len'].append(len(raw_info['tokens']))
            if sent_id not in noms_dict:
                cur_nom_instance['nominals_len'].append(0)
            else:
                nom_info = noms_dict[sent_id]
                cur_nom_instance['nominals'] += nom_info['nominals']
                cur_nom_instance['nominals_len'] += nom_info['nominals_len']
                cur_nom_instance['nominal_indices'] += [list(map(int, prior_nom_word_counts+np.array(i))) for i in nom_info['nominal_indices']]

            prior_verb_word_counts = len(cur_verb_instance['words'])
            cur_verb_instance['words'] += raw_info['tokens']
            cur_verb_instance['sent_len'].append(len(raw_info['tokens']))
            if sent_id not in verbs_dict:
                cur_verb_instance['verbs_len'].append(0)
            else:
                verb_info = verbs_dict[sent_id]
                cur_verb_instance['verbs'] += verb_info['verbs']
                cur_verb_instance['verbs_len'] += verb_info['verbs_len']
                cur_verb_instance['verb_index'] += [prior_verb_word_counts + i for i in verb_info['verb_index']]

        prior_doc_id = doc_id
        prior_sent_num = sent_num
    if cur_nom_instance['doc_id'] != nom_sample_list[-1]['doc_id']:
        nom_sample_list.append(cur_nom_instance)
    if cur_verb_instance['doc_id'] != verb_sample_list[-1]['doc_id']:
        verb_sample_list.append(cur_verb_instance)
    assert len(nom_sample_list) == len(verb_sample_list) == len(set(doc_id_list))

    fnom_out = open(nom_doc_path, 'w')
    fverb_out = open(verb_doc_path, 'w')
    for idx, (nom_sample, verb_sample) in tqdm(enumerate(zip(nom_sample_list, verb_sample_list)), total=len(verb_sample_list)):
        fnom_out.write(json.dumps(nom_sample)+'\n')
        fverb_out.write(json.dumps(verb_sample)+'\n')
    fnom_out.close()
    fverb_out.close()

    return

def preserve_events(corpus_path, verb_path, nom_path, saved_verb_path, saved_nom_path):
    raw_sent_id_info_mapping = dict()
    for raw_line in open(corpus_path):
        raw_data = json.loads(raw_line)
        raw_sent_id_info_mapping[raw_data['sent_id']] = deepcopy(raw_data)

    num_raw = sum(1 for line in open(corpus_path))
    num_nom, num_verb = sum(1 for i in open(nom_path)), sum(1 for i in open(verb_path))
    num_tokens = sum(1 for i in open(corpus_path))
    assert num_raw == num_nom == num_verb == num_tokens
    saved_verb, saved_nom = list(), list()
    verb_counts, nom_counts = 0, 0
    event_types, total_event_types = list(), list()
    for line_idx, (token_line, nom_line, verb_line) in tqdm(enumerate(zip(open(corpus_path), open(nom_path), open(verb_path))), total=num_raw):
        token_info, nom_info, verb_info = json.loads(token_line), json.loads(nom_line), json.loads(verb_line)
        raw_info = raw_sent_id_info_mapping[token_info['sent_id']]
        assert raw_info['tokens'] == nom_info['words'] == verb_info['words']

        covered_evt_idx_list = list()
        for verb_data in verb_info['verbs']:
            text, index = verb_data['verb'], verb_data['verb_index']
            if index not in covered_evt_idx_list:
                cur_saved_verb = deepcopy(verb_data)
                cur_saved_verb['words'] = verb_info['words']
                cur_saved_verb['line_idx'] = line_idx
                saved_verb.append(cur_saved_verb)
                covered_evt_idx_list.append(index)
                verb_counts += 1
        for nom_data in nom_info['nominals']:
            text, nominal_indices = nom_data['nominal'], nom_data['nominal_indices']
            for nominal_index in nominal_indices:
                if nominal_index not in covered_evt_idx_list:
                    cur_saved_nom = deepcopy(nom_data)
                    cur_saved_nom['words'] = nom_info['words']
                    cur_saved_nom['line_idx'] = line_idx
                    saved_nom.append(cur_saved_nom)
                    covered_evt_idx_list.append(nominal_index)
                    nom_counts += 1
    logger.info(f'verb_counts: {verb_counts} ({verb_counts}), nom_counts: {nom_counts} ({nom_counts})')
    logger.info(f'event_types: {len(set(event_types))}, total: {len(set(total_event_types))}')

    # save
    fverb_out = open(saved_verb_path, 'w')
    for i in tqdm(saved_verb):
        fverb_out.write(json.dumps(i)+'\n')
    fverb_out.close()
    logger.info(f'{len(saved_verb)} verbal events saved...')
    # 790 verbal events saved...
    fnom_out = open(saved_nom_path, 'w')
    for i in tqdm(saved_nom):
        fnom_out.write(json.dumps(i)+'\n')
    fnom_out.close()
    logger.info(f'{len(saved_nom)} nominal events saved...')
    # 668 nominal events saved...
    return

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sent_sbert_repre(token_path, saved_folder, gpu_id, batch_size=4, model_name='sentence-transformers/all-roberta-large-v1', overwrite=False):
    os.makedirs(saved_folder, exist_ok=True)
    if os.path.exists(os.path.join(saved_folder, f'sentence_sbert_emb.pkl')) and not overwrite:
        logger.info(f'find results at {os.path.join(saved_folder, "sentence_sbert_emb.pkl")}...')
        return

    device = torch.device(f'cuda:{gpu_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model = model.to(device)

    def encode(example):
        tokenized_inputs = tokenizer(example['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
        return tokenized_inputs
    # {"sentence": " prompt"}
    dataset = load_dataset('json',
                           data_files=token_path,
                           split='train')
    dataset = dataset.map(encode, batched=True)
    dataset = dataset.remove_columns(column_names=list(set(dataset.column_names) - set(inference_include_columns_dict[model_name])))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    results = list()
    for i, batch in enumerate(tqdm(dataloader)):
        new_batch = {k: torch.stack(batch[k], dim=-1).to(device) for k in inference_include_columns_dict[model_name]}
        # [batch_size, seqlen, hidden_dims]
        model_output = model(**new_batch)
        sentence_embeddings = _mean_pooling(model_output, new_batch['attention_mask'])
        outputs = F.normalize(sentence_embeddings, p=2, dim=1).detach().cpu().numpy()
        for output in outputs:
            results.append({'sbert_emb': output})
    with open(os.path.join(saved_folder, 'sentence_sbert_emb.pkl'), 'wb') as f:
        pickle.dump(results, f)
    return

def get_token_repre(token_path, saved_folder, gpu_id,
                        batch_size=10, model_name='bert-large-uncased-whole-word-masking', overwrite=False):
    os.makedirs(saved_folder, exist_ok=True)
    if os.path.exists(os.path.join(saved_folder, f'sent_last_hidden_states.pkl')) and not overwrite:
        logger.info(f'find files at {os.path.join(saved_folder, f"sent_last_hidden_states.pkl")}...')
        return

    device = torch.device(f'cuda:{gpu_id}')
    model_class = BertForMaskedLM

    mlm_model = model_class.from_pretrained(model_name)
    mlm_model.eval()
    mlm_model = mlm_model.to(device)

    # get model for embedding extraction
    model = mlm_model.bert
    # NOTE: interesting, specified tokenizer does not support word_ids, but autotokenizer supports
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(example):
        tokenized_inputs = tokenizer(
            example['tokens'],
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
        )
        token_word_ids = list()
        for i in range(len(example['tokens'])):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            token_word_ids.append(word_ids)
        tokenized_inputs['token_word_ids'] = token_word_ids
        return tokenized_inputs

    def collate_word(data):
        output = dict()
        for element in data:
            for key, val in element.items():
                if key not in output:
                    output[key] = [val]
                else:
                    output[key].append(val)
        finalized_output = dict()
        for key, val in output.items():
            if key in inference_include_columns_dict[model_name]:
                finalized_output[key] = torch.from_numpy(np.array(val))
            else:
                finalized_output[key] = val

        return finalized_output

    dataset = load_dataset('json',
                           data_files=token_path,
                           split='train')
    dataset = dataset.map(encode, batched=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_word)

    results = list()
    for i, batch in enumerate(tqdm(dataloader)):
        new_batch = {k: batch[k].to(device) for k in inference_include_columns_dict[model_name]}
        # [batch_size, seqlen, hidden_dims]
        outputs = model(**new_batch).last_hidden_state.detach().cpu().numpy()
        for idx, (output, token_word_ids) in enumerate(zip(outputs, batch['token_word_ids'])):
            seqlen = sum(batch['input_ids'][idx] != tokenizer.pad_token_id)
            results.append({'last_hidden_states': output[:seqlen, :], 'token_word_ids': token_word_ids[:seqlen]})
    with open(os.path.join(saved_folder, 'sent_last_hidden_states.pkl'), 'wb') as f:
        pickle.dump(results, f)
    return

def extract_event_repre(source_folder, overwrite, sense_srl_file_, source_list=['verb', 'nom']):
    sentence_sbert_emb_path = os.path.join(source_folder, 'sentence_transformers', 'sentence_sbert_emb.pkl')

    if os.path.exists(os.path.join(source_folder, f'verb_arg_sense_emb.pkl')) and os.path.exists(os.path.join(source_folder, f'nom_arg_sense_emb.pkl')) and not overwrite:
        logger.info(f'find verb argument representation at {os.path.join(source_folder, f"verb/nom_arg_sense_emb.pkl")}...')
        return

    with open(os.path.join(source_folder, 'bert_large_repre', 'sent_last_hidden_states.pkl'), 'rb') as f:
        last_hidden_states = pickle.load(f)
    with open(sentence_sbert_emb_path, 'rb') as f:
        sbert_emb = pickle.load(f)

    reference = list()
    assert len(last_hidden_states) == len(sbert_emb)
    for hidden_info, sbert_info in tqdm(zip(last_hidden_states, sbert_emb), total=len(last_hidden_states)):
        word_token_ids = dict()
        for token_id, word_id in enumerate(hidden_info['token_word_ids']):
            if word_id in word_token_ids:
                word_token_ids[word_id].append(token_id)
            else:
                word_token_ids[word_id] = [token_id]
        reference.append({
            'last_hidden_states': hidden_info['last_hidden_states'],
            'word_token_ids': word_token_ids,
            'token_word_ids': hidden_info['token_word_ids'],
            'sbert_emb': sbert_info['sbert_emb'],
        })

    # verb
    for source in source_list:
        saved_path = os.path.join(source_folder, f'{source}_arg_sense_emb.pkl')
        if os.path.exists(saved_path) and not overwrite:
            continue
        sense_info = pickle.load(open(sense_path[source], 'rb'))
        saved_results = list()
        sense_srl_file = sense_srl_file_.format(source)
        total_lines = sum(1 for i in open(sense_srl_file))
        for idx, line in tqdm(enumerate(open(sense_srl_file)), total=total_lines):
            predicate_info = json.loads(line)
            line_idx = predicate_info['line_idx']
            hidden_repre, word_token_ids, token_word_ids = reference[line_idx]['last_hidden_states'], reference[line_idx]['word_token_ids'], reference[line_idx]['token_word_ids']
            sent_sbert_emb = reference[line_idx]['sbert_emb']
            assert np.max(token_word_ids[1:-1]) + 1 == len(predicate_info['words'])

            lemma_pos, sense_explanation_emb, sense_example_emb, arg1_repre, arg2_repre = None, None, None, None, None

            # 1. predicate sense definition and example representation
            doc = nlp(predicate_info['words'])
            if source == 'verb':
                predicate_lemma_list = [doc[predicate_info['verb_index']].lemma_]
            else:
                predicate_lemma_list = [doc[i].lemma_ for i in predicate_info['nominal_indices']]
            sense_explanation_emb, sense_example_emb = None, None
            for predicate_lemma in predicate_lemma_list:
                if f'{predicate_lemma}_v' in sense_info:
                    lemma_pos = f'{predicate_lemma}_v'
                elif f'{predicate_lemma}_n' in sense_info:
                    lemma_pos = f'{predicate_lemma}_n'
                elif predicate_lemma in sense_info:
                    lemma_pos = predicate_lemma
                if lemma_pos is not None:
                    sense_id = predicate_info['sense']
                    if source == 'verb' and np.abs(eval(sense_id) - int(eval(sense_id))) < 1e-10:
                        sense_id = str(int(eval(sense_id)))
                    try:
                        extract_sense_info = sense_info[lemma_pos][sense_id]
                        sense_explanation_emb = extract_sense_info['explanation_emb']
                        if type(sense_explanation_emb) == torch.Tensor:
                            sense_explanation_emb = sense_explanation_emb.numpy()
                        sense_example_emb = extract_sense_info['example_emb']
                        break
                    except:
                        logger.info(f'{sense_id} not in {list(sense_info[lemma_pos].keys())} for {lemma_pos}')
            # 2. predicate token representation
            if source == 'verb':
                matched_token_ids = np.array(word_token_ids[predicate_info['verb_index']])
            else:
                matched_token_ids = np.concatenate([np.array(word_token_ids[i]) for i in predicate_info['nominal_indices']], axis=0)
            predicate_repre = np.mean(hidden_repre[matched_token_ids], axis=0)
            # 3. arg1 and arg2 representation
            arg1_index, arg1_lemma, arg2_index, arg2_lemma = list(), list(), list(), list()
            for idx, tag in enumerate(predicate_info['tags']):
                if tag.endswith('ARG1'):
                    arg1_index.append(idx)
                    arg1_lemma.append(doc[idx].lemma_)
                elif tag.endswith('ARG2'):
                    arg2_index.append(idx)
                    arg2_lemma.append(doc[idx].lemma_)
            if len(arg1_index) > 0:
                arg1_repre = np.mean(hidden_repre[np.array(arg1_index)], axis=0)
            if len(arg2_index) > 0:
                arg2_repre = np.mean(hidden_repre[np.array(arg2_index)], axis=0)

            # summarize all information
            cur_saved_info = deepcopy(predicate_info)
            cur_saved_info['predicate_emb'] = predicate_repre
            cur_saved_info['predicate_lemma'] = predicate_lemma_list
            cur_saved_info['arg1_emb'] = arg1_repre
            cur_saved_info['arg1_lemma'] = arg1_lemma
            cur_saved_info['arg2_emb'] = arg2_repre
            cur_saved_info['arg2_lemma'] = arg2_lemma
            cur_saved_info['sent_sbert_emb'] = sent_sbert_emb
            cur_saved_info['sense_explanation_emb'] = sense_explanation_emb
            cur_saved_info['sense_example_emb'] = sense_example_emb
            saved_results.append(cur_saved_info)
        logger.info(f'{source} saved_results: {len(saved_results)}')
        with open(saved_path, 'wb') as f:
            pickle.dump(saved_results, f)
    return

def combine_plain_embeddings(feature_saved_path, overwrite):
    parent_folder = os.path.dirname(feature_saved_path)
    emb_source = ['predicate_emb', 'arg1_emb', 'arg2_emb', 'sent_sbert_emb', 'sense_explanation_emb', 'sense_example_emb']
    emb_merge_format = 'average'
    emb_source = sorted(emb_source)
    emb_source_name = '_'.join(emb_source+[emb_merge_format])

    data_saved_path = f'{parent_folder}/{emb_source_name}/samples.pkl'
    os.makedirs(os.path.dirname(data_saved_path), exist_ok=True)
    logger.info('step I: extract features')
    if not os.path.exists(data_saved_path) or overwrite:
        X, source_name, source_idx = list(), list(), list()
        for predicate_type in ['verb', 'nom']:
            samples = pickle.load(open(f'{parent_folder}/{predicate_type}_arg_sense_emb.pkl', 'rb'))
            for local_idx, sample in tqdm(enumerate(samples), total=len(samples)):
                source_name.append(predicate_type)
                source_idx.append(local_idx)
                cur_X = list()
                for one_source in emb_source:
                    cur_emb = sample[one_source]
                    if cur_emb is None:
                        continue
                    assert cur_emb.dtype==np.float32 and np.sum(np.isnan(cur_emb)) == 0
                    cur_X.append(cur_emb)
                assert len(cur_X) > 0
                if emb_merge_format == 'average':
                    X.append(np.mean(np.stack(cur_X, axis=0), axis=0))
                elif emb_merge_format == 'concat':
                    X.append(np.concatenate(cur_X))
        X = np.stack(X, axis=0)
        with open(data_saved_path, 'wb') as f:
            pickle.dump({
                'X': X,
                'source_name': source_name,
                'source_idx': source_idx
            }, f)

        logger.info(f'Samples of shape {X.shape} saved at {data_saved_path}')

    info = pickle.load(open(data_saved_path, 'rb'))
    fout = open(feature_saved_path, 'w')
    for id, x in tqdm(enumerate(info['X'])):
      x_str = '\t'.join(list(map(str, x)))
      fout.write(str(id))
      fout.write('\t')
      fout.write(str(0)) # 0 is placeholder
      fout.write('\t')
      fout.write(x_str)
      fout.write('\n')
    fout.close()
    logger.info(f'generate tsv samples at {feature_saved_path}')
    return

def generate_linkage_ward_tree(pred_path, data_path, random_seed, overwrite):
    if os.path.exists(pred_path) and not overwrite:
        return
    _set_random_seed(random_seed)

    raw = np.loadtxt(data_path, dtype=np.float32)
    y = raw[:, 1].astype(np.int32) # y is a placeholder, 0 in feature files
    X = raw[:, 2:]

    Z = scipy_linkage(X, 'ward')

    fout = open(pred_path, 'w')

    fout_info = dict()

    root_candidates, child_list = list(), list()
    for idx, cluster_info in enumerate(Z):
        parent_idx = len(X) + idx
        left_child_id, right_child_id = int(cluster_info[0]), int(cluster_info[1])
        left_child_label, right_child_label = None, None
        if left_child_id < len(X):
            left_child_label = y[left_child_id]
        if right_child_id < len(X):
            right_child_label = y[right_child_id]
        fout_info[left_child_id] = ["%s\t%s\t%s\n" % (left_child_id, parent_idx, left_child_label), parent_idx, left_child_label]

        fout_info[right_child_id] = ["%s\t%s\t%s\n" % (right_child_id, parent_idx, right_child_label), parent_idx, right_child_label]

        root_candidates.append(parent_idx)
        child_list.append(left_child_id)
        child_list.append(right_child_id)
    root_idx = list(set(root_candidates) - set(child_list))
    assert len(root_idx) <= 1
    if len(root_idx) == 1:
        fout_info[root_idx[0]] = ["%s\t%s\t%s\n" % (root_idx[0], None, None), None, None]
    to_save_id = list()
    for idx, x in enumerate(X):
        to_save_id.append(idx)
        parent = fout_info[idx][1]
        while parent is not None:
            to_save_id.append(parent)
            parent = fout_info[parent][1]

    to_save_id = list(set(to_save_id))
    for child_id in to_save_id:
        fout.write(fout_info[child_id][0])

    fout.close()

    return

