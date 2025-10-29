import argparse
import wandb
import torch
import os
import random
import math

import numpy as np

from tqdm import tqdm
from dotenv import load_dotenv
from cs336_basics.transformer import Transformer
from cs336_basics.utils import AdamW
from cs336_basics.utils import data_loading, cross_entropy
from cs336_basics.utils import lr_cosine_sheduling, gradient_clipping
from cs336_basics.utils import save_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def init_wandb(args):
    load_dotenv()

    wandb_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_key)

    # Initialize a new run
    wandb.init(
        project="cs336-assign1", 
        name=args.dataset_name)

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")

def train_loop(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_data = np.load(args.train_path, mmap_mode='r')
    valid_data = np.load(args.valid_path, mmap_mode='r')

    num_train_batches = math.ceil(len(train_data) / args.batch_size)
    total_train_iters = args.num_epochs * num_train_batches
    pbar = tqdm(total=total_train_iters, desc="train")
    warmup_iters = int(total_train_iters * args.warmup_ratio)
    cosine_cycle_iters = total_train_iters - warmup_iters

    model = Transformer(vocab_size=args.vocab_size, 
                        context_length=args.context_length,
                        num_layers=args.num_layers,
                        d_model=args.d_model,
                        num_heads=args.num_heads,
                        d_ff=args.d_ff,
                        rope_theta=args.rope_theta,
                        device=device)
    model.to(device)
    optimizer = AdamW(model.parameters(), 
                      lr=args.max_lr, 
                      weight_decay=args.weight_decay,
                      betas=tuple(args.betas),
                      eps=args.eps)
    
    min_valid_loss = math.inf
    global_step = 0

    for _ in range(args.num_epochs):
        for train_iter in range(num_train_batches):
            global_step += 1

            model.train()
            inputs, targets = data_loading(
                train_data[train_iter*args.batch_size: (train_iter+1)*args.batch_size],
                args.batch_size, args.context_length, device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm=args.max_l2_norm)
            
            new_lr = lr_cosine_sheduling(
                global_step, args.max_lr, args.min_lr, 
                warmup_iters, cosine_cycle_iters)
            for group in optimizer.param_groups:
                group["lr"] = new_lr
            optimizer.step()
            
            if global_step % args.valid_interval_iters == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss_list = []
                    
                    for valid_iter in range(math.ceil(len(valid_data) / args.batch_size)):
                        valid_inputs, valid_targets = data_loading(
                            valid_data[valid_iter*args.batch_size: (valid_iter+1)*args.batch_size],
                            args.batch_size, args.context_length, device
                        )
                        valid_loss_list.append(cross_entropy(model(valid_inputs), valid_targets).item())
                    
                    mean_valid_loss = np.mean(valid_loss_list)
                    wandb.log({
                        "eval/loss": mean_valid_loss
                    }, step=global_step)
                    if mean_valid_loss < min_valid_loss:
                        min_valid_loss = mean_valid_loss
                        saved_path = os.path.join(
                            args.checkpoint_path, 
                            args.dataset_name, 
                            f"checkpoint_batch{args.batch_size}_maxlr{args.max_lr}_iter{global_step}_loss{mean_valid_loss:.4f}.pth")
                        save_checkpoint(model, optimizer, 
                                        out=saved_path)   
                        print(f"iter = {global_step}, loss = {mean_valid_loss:.4f}, checkpoint saved!")

            pbar.update(1)
            loss_item = loss.item()
            pbar.set_postfix(
                loss=f"{loss_item:.4f}",
                lr=f"{new_lr:.6f}"
            )
            wandb.log({
                "global_step": global_step,
                "train/loss": loss_item,
                "train/lr": new_lr
            }, step=global_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint")
    parser.add_argument("--train_path", type=str, default="data/TinyStoriesV2-GPT4-train-encoding.npy")
    parser.add_argument("--valid_path", type=str, default="data/TinyStoriesV2-GPT4-valid-encoding.npy")
    parser.add_argument("--dataset_name", type=str, default="TinyStoriesV2")

    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.98))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--valid_interval_iters", type=int, default=10000)

    args = parser.parse_args()

    set_seed(args.seed)
    init_wandb(args)

    train_loop(args)
    wandb.finish()