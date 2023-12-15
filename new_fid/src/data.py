import torch
import random
import json
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 bug_prefix='bug:',
                 error_prefix='err:',
                 code_context_prefix='ctxs:'):
        self.data = data
        self.n_context = n_context
        self.bug_line_prefix = bug_prefix
        self.error_prefix = error_prefix
        self.code_context_prefix = code_context_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'fix' in example:
            target = example['fix']
            return target + ' </s>'
        elif 'fixes' in example:
            return random.choice(example['fixes']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        bug_line = self.bug_line_prefix + " " + example['bug']
        error = self.error_prefix + " " + example['error']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            # f = self.title_prefix + " {} " + self.code_context_prefix + " {}"
            f = self.code_context_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            # code_contexts = [f.format(c['title'], c['text']) for c in contexts]
            code_contexts = [f.format(c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [bug_line]
        else:
            code_contexts, scores = None, None


        return {
            'index' : index,
            'bug' : bug_line,
            'target' : target,
            'error' :error,
            'contexts': code_contexts,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def encode_context(batch_code_context, tokenizer, max_length):
    context_ids, context_mask = [], []
    for k, text_contexts in enumerate(batch_code_context):
        p = tokenizer.batch_encode_plus(
            text_contexts,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        context_ids.append(p['input_ids'][None])
        context_mask.append(p['attention_mask'][None])

    context_ids = torch.cat(context_ids, dim=0)
    context_mask = torch.cat(context_mask, dim=0)
    return context_ids, context_mask.bool()

class Collator(object):
    def __init__(self, text_max_length, tokenizer, fix_max_length=512):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.fix_max_length = fix_max_length

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.fix_max_length if self.fix_max_length > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.fix_max_length > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_bug(example):
            if example['contexts'] is None:
                print("no context check this out data.py")
                return [example['bug']]
            return [example['bug'] + " " + t for t in example['contexts']]
        text_context = [append_bug(example) for example in batch]
        context_ids, context_masks = encode_context(text_context,
                                                     self.tokenizer,
                                                     self.fix_max_length)

        return (index, target_ids, target_mask, context_ids, context_masks)

def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples

class RetrieverCollator(object):
    def __init__(self, tokenizer, context_max_length=450, bug_max_length=62):
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.bug_max_length = bug_max_length

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        bug = [ex['bug'] for ex in batch]
        bug = self.tokenizer.batch_encode_plus(
            bug,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.bug_max_length,
            truncation=True
        )
        bug_ids = bug['input_ids']
        bug_mask = bug['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['contexts'] is None:
            return index, bug_ids, bug_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        contexts = [ex['contexts'] for ex in batch]
        context_ids, context_masks = encode_context(
            contexts,
            self.tokenizer,
            self.context_max_length
        )

        return (index, bug_ids, bug_mask, context_ids, context_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                #  title_prefix='title:',
                 error_prefix='err:',
                 context_prefix='context:'):
        self.data = data
        # self.title_prefix = title_prefix
        self.context_prefix = context_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        # text = self.title_prefix + " " + example[2] + " " + \
        #     self.context_prefix + " " + example[1]
        text = self.context_prefix + " " + example[1] 
        # + " " + \ self.error_prefix + " " + example[2]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, max_length=450):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
