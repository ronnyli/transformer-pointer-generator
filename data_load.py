# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''

import glob
import json
import tensorflow as tf
import torch
from utils import calc_num_batches

def _load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    vocab = []
    with open(vocab_fpath, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.replace('\n', ''))
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}

    return token2idx, idx2token

def load_stop(vocab_path):
    """
    load stop word
    :param vocab_path: stop word path
    :return: stop word list
    """
    stop_words = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words.append(line.replace('\n', ''))

    return sorted(stop_words, key=lambda i: len(i), reverse=True)

def _load_json(txt, source_key='input', target_key='target'):
    json_obj = json.loads(txt)
    return json_obj[source_key], json_obj[target_key]

def _load_csv(txt):
    splits = line.split(',')
    # if len(splits) != 2: continue
    sen1 = splits[1].replace('\n', '').strip()
    sen2 = splits[0].strip()
    return sen1, sen2

def _load_pt(fpaths, maxlen1, maxlen2, is_chinese=True):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    n = 0
    for fpath in glob.glob(fpaths):
        assert fpath.endswith('.pt'), f'Not a PyTorch file: {fpath}'
        if n >= 40000:
            break
        f = torch.load(fpath)
        for line in f:
            n += 1
            if n >= 40000:
                break
            sen1 = line['src_txt']
            sen2 = line['tgt_txt']
            if is_chinese:
                sen1_tokens = list(sen1)
                sen2_tokens = list(sen2)
                sen1 = sen1.encode('utf-8')
                sen2 = sen2.encode('utf-8')
                if len(sen1_tokens) + 1 > maxlen1-2: continue
                if len(sen2_tokens) + 1 > maxlen2-1: continue
            else:
                # TODO: check for better tokenizer
                sen1_tokens = ' '.join(sen1)
                sen1_tokens = sen1_tokens.split(' ')
                sen2_tokens = sen2.replace('<q>', ' ').split(' ')
                if len(sen2_tokens) + 1 > maxlen2-1: continue  # HACK: too lazy to figure out where the <q> tags were previously
                if len(sen1_tokens) + 1 > maxlen1-2:
                    sen1 = ' '.join(sen1_tokens[:(maxlen1 - 3)])
            sents1.append(sen1)
            sents2.append(sen2)
    return sents1, sents2


def _load_data(fpaths, maxlen1, maxlen2):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    for fpath in fpaths.split('|'):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                sen1, sen2 = _load_csv(line)
                if len(list(sen1)) + 1 > maxlen1-2: continue
                if len(list(sen2)) + 1 > maxlen2-1: continue

                sents1.append(sen1.encode('utf-8'))
                sents2.append(sen2.encode('utf-8'))

    return sents1[:400000], sents2[:400000]

def _encode(inp, token2idx, maxlen, type, is_chinese=True):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    inp = inp.decode('utf-8')
    if is_chinese:
        inp_tokens = list(inp)
    else:
        inp_tokens = inp.split(' ')
    if type == 'x':
        tokens = ['<s>'] + inp_tokens + ['</s>']
        while len(tokens) < maxlen:
            tokens.append('<pad>')
        return [token2idx.get(token, token2idx['<unk>']) for token in tokens]

    else:
        inputs = ['<s>'] + inp_tokens
        target = inp_tokens + ['</s>']
        while len(target) < maxlen:
            inputs.append('<pad>')
            target.append('<pad>')
        return [token2idx.get(token, token2idx['<unk>']) for token in inputs], [token2idx.get(token, token2idx['<unk>']) for token in target]

def _generator_fn(sents1, sents2, vocab_fpath, maxlen1, maxlen2):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _ = _load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        x = _encode(sent1, token2idx, maxlen1, "x", is_chinese=False)  # TODO: is_chinese should be param

        inputs, targets = _encode(sent2, token2idx, maxlen2, "y", is_chinese=False)  # TODO: is_chinese should be param

        yield (x, sent1.decode('utf-8')), (inputs, targets, sent2.decode('utf-8'))

def _input_fn(sents1, sents2, vocab_fpath, batch_size, gpu_nums, maxlen1, maxlen2, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    shapes = (([maxlen1], ()),
              ([maxlen2], [maxlen2], ()))
    types = ((tf.int32, tf.string),
             (tf.int32, tf.int32, tf.string))

    dataset = tf.data.Dataset.from_generator(
        _generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath, maxlen1, maxlen2))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size*gpu_nums)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.batch(batch_size*gpu_nums)

    return dataset

def get_batch(fpath, maxlen1, maxlen2, vocab_fpath, batch_size, gpu_nums, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath: source file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    sents1, sents2 = _load_pt(fpath, maxlen1, maxlen2, is_chinese=False)  # TODO: is_chinese should be command line parameter
    batches = _input_fn(sents1, sents2, vocab_fpath, batch_size, gpu_nums, maxlen1, maxlen2, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size*gpu_nums)
    return batches, num_batches, len(sents1)
