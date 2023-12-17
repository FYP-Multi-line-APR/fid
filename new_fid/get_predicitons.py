# get_predictions.py

import torch
import transformers
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
from src.meta import MODEL_NAME
from src.meta import NO_WORKERS

import src.slurm
import src.util
from src.options import Options
import src.data
import src.model
import src.evaluation

def get_predictions(model, dataset, dataloader, tokenizer, opt):
    predictions = []
    model.eval()
    if hasattr(model, "module"):
        model = model.module

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                predictions.append((idx[k], ans))

    return predictions

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, return_dict=False)

    collator_function = src.data.Collator(opt.text_max_length, tokenizer)
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples,
        opt.n_context,
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=NO_WORKERS,
        collate_fn=collator_function
    )

    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    logger.info("Start getting predictions")

    predictions = get_predictions(model, eval_dataset, eval_dataloader, tokenizer, opt)

    # Save predictions to a file
    output_file = Path(opt.checkpoint_dir) / opt.name / 'predictions.txt'
    with open(output_file, 'w') as f:
        for idx, prediction in predictions:
            f.write(f"{idx}\t{prediction}\n")

    logger.info(f'Predictions saved to {output_file}')
