from torch.utils.tensorboard import SummaryWriter
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

def trace_wrapper(model, context_ids, context_mask, decoder_input_ids, decoder_mask):
    model.eval()
    with torch.no_grad():
        return model(
            input_ids=context_ids,
            attention_mask=context_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_mask
        )

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    dir_path = Path(opt.checkpoint_dir) / opt.name
    directory_exists = dir_path.exists()

    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'prediction_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

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

    writer = SummaryWriter()

    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            # Move tensors to the same device as the model
            context_ids = context_ids.to(opt.device)
            context_mask = context_mask.to(opt.device)

            # Example: You can use decoder_input_ids or decoder_inputs_embeds based on your use case
            decoder_input_ids = torch.zeros_like(context_ids)  # Replace this with your actual input
            decoder_attention_mask = torch.ones_like(context_mask)  # Replace this with your actual attention mask

            # Wrap the model with the trace_wrapper
            traced_model = torch.jit.trace(
                trace_wrapper, (model, context_ids, context_mask, decoder_input_ids, decoder_attention_mask)
            )

            # Add the traced model to TensorBoard
            writer.add_graph(traced_model)

            # Rest of your evaluation code...
            break
