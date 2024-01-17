

import os

import argparse
import csv
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers

import src.model
import src.data
import src.util
import src.slurm 


logger = logging.getLogger(__name__)

def embed_contexts(opt, contexts, model, tokenizer):
    batch_size = opt.per_gpu_batch_size * opt.world_size
    # FIXME: model.config.passage_maxlength is not defined that should be kind of context_max_lenght
    collator = src.data.TextCollator(tokenizer, model.config.context_max_length)
    dataset = src.data.TextDataset(contexts, error_prefix='err:', context_prefix='context:')
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(dataloader):
            embeddings = model.embed_text(
                text_ids=text_ids.cuda(), 
                text_mask=text_mask.cuda(), 
                # FIXME: model.config.apply_passage_mask is not defined this should be kind of apply_context_mask
                apply_mask=model.config.apply_passage_mask,
                extract_cls=model.config.extract_cls,
            )
            embeddings = embeddings.cpu()
            total += len(ids)

            allids.append(ids)
            allembeddings.append(embeddings)
            if k % 100 == 0:
                logger.info('Encoded passages %d', total)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = [x for idlist in allids for x in idlist]
    return allids, allembeddings


def main(opt):
    logger = src.util.init_logger(is_main=True)
    tokenizer = transformers.RobertaTokenizer.from_pretrained(src.meta.MODEL_NAME)
    model_class = src.model.Retriever
    #model, _, _, _, _, _ = src.util.load(model_class, opt.model_path, opt)
    model = model_class.from_pretrained(opt.model_path)
    
    model.eval()
    model = model.to(opt.device)
    if not opt.no_fp16:
        model = model.half()

    # passages = src.util.load_passages(args.passages)
    contexts = src.util.load_contexts(args.contexts)

    shard_size = len(contexts) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards-1:
        end_idx = len(contexts)

    contexts = contexts[start_idx:end_idx]
    logger.info(f'Embedding generation for {len(contexts)} contexts from idx {start_idx} to {end_idx}')

    allids, allembeddings = embed_contexts(opt, contexts, model, tokenizer)

    output_path = Path(args.output_path)
    save_file = output_path.parent / (output_path.name + f'_{args.shard_id:02d}')
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    logger.info(f'Saving {len(allids)} context embeddings to {save_file}')
    with open(save_file, mode='wb') as f:
        pickle.dump((allids, allembeddings), f)

    logger.info(f'Total context processed {len(allids)}. Written to {save_file}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--contexts', type=str, default=None, help='Path to contexts (.tsv file)')
    parser.add_argument('--output_path', type=str, default='code_context_embeddings/contexts', help='prefix path to save embeddings')
    parser.add_argument('--shard_id', type=int, default=0, help="Id of the current shard")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of shards")
    parser.add_argument('--per_gpu_batch_size', type=int, default=16, help="Batch size for the context encoder forward pass")
    parser.add_argument('--context_max_length', type=int, default=450, help="Maximum number of tokens in a context")
    parser.add_argument('--model_path', type=str, help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    args = parser.parse_args()

    src.slurm.init_distributed_mode(args)

    main(args)
