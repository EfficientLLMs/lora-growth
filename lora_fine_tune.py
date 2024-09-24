import argparse
import torch
import utils
from accelerate import Accelerator
import wandb
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from lora_weight_tying import LoraWeightTyingConfig, get_peft_model_weight_tying



if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='1.4b')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="alpaca_gpt4", choices=["alpaca_gpt4", "gsm8k"])
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    args.wandb_name = f"{args.name}_r={args.rank}_{args.lr}_{args.dataset}"
    args.output = f"./models/raw/pythia_{args.name}_r={args.rank}_{args.lr}_{args.dataset}/"

    # define the base model
    base_model_name = f"EleutherAI/pythia-{args.name}"

    # accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"device: {device}")

    # seed
    utils.seed_everything(args.seed)

    # dataloader
    train_dataloader, eval_dataloader = utils.get_dataloader(args.dataset, args.batch_size)

    # wandb
    if accelerator.is_main_process:
        wandb.init(
            name=args.wandb_name,
            project="lora-finetune-tying", 
            entity="vibhamasti"
        )

        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": train_dataloader.batch_size,
            "seed": args.seed,
            "dataset": args.dataset,
            "rank": args.rank
        }

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,  # standard model; the same tokenizer is used for all models
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # base model
    model = GPTNeoXForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        use_cache=False,
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # config for weight tying
    config = LoraWeightTyingConfig(
        r=args.rank, 
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query_key_value"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # get the PEFT model with weight tying
    model = get_peft_model_weight_tying(model, config)

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # prepare for accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    scheduler = None

    # train/evaluate and save the model
    utils.train_model(
        model, 
        optimizer, 
        scheduler, 
        args.lr, 
        train_dataloader, 
        eval_dataloader, 
        args.epochs, 
        accelerator, 
        tokenizer, 
        args.output
    )
    wandb.finish()




