import os
import sys
import json
import torch
import random
import datetime
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
#from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
from models.fid_gr_modules import FiDGRModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args, train_params):
    sys.setrecursionlimit(10000)
    set_seed(args.seed)
    model = FiDGRModel(args)
    trainer = pl.Trainer(**train_params)
    if args.do_train:
        now = datetime.datetime.now()
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Start Training...")
        if args.resume_from_checkpoint is None:
            trainer.fit(model)
        else:
            print(f"@@@ Resume Training from {args.resume_from_checkpoint}")
            trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
        now = datetime.datetime.now()
        print(
            f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Done Training..."
        )
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_model", default='t5-base', type=str)
    parser.add_argument('--num_workers', default=64, type=int)
    parser.add_argument("--machine", default='lg', type=str)
    parser.add_argument("--exp_mode", default='train', type=str)
    parser.add_argument("--eval_mode", default='1', type=str)
    parser.add_argument("--test_model_path", default='', type=str) # You may just write the directory and not the exact tfmr num. We'll find the correct tfmr for you.
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    parser.add_argument("--from_model_path", default=None, type=str)
    parser.add_argument("--dataset", type=str, default='nq')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--run_3b", action='store_true')
    parser.add_argument("--load_from_fid", action='store_true')
    parser.add_argument("--learning_rate", default=6e-05, type=float)
    parser.add_argument("--train_batch_size", default=48, type=int)
    parser.add_argument("--eval_batch_size", default=48, type=int)
    parser.add_argument("--val_beam_size", default=4, type=int)
    parser.add_argument("--max_input_length", default=512, type=int)
    parser.add_argument("--max_output_length", default=4, type=int)
    # file names
    parser.add_argument("--train-files", default=['../dataset/downloads/nq-train-small.json'], nargs='+')
    parser.add_argument("--eval-files", default=['../dataset/downloads/nq-dev.json'], nargs='+')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument("--sub-mode", default='', type=str)
    parser.add_argument("--prompt_type", default='2', type=str)
    parser.add_argument("--options", default='None', type=str)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--name', default='noname', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--lr_scheduler', default='constant', type=str)
    parser.add_argument('--accelerator', default='deepspeed', type=str)
    parser.add_argument('--eval_steps', default=3000, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--num_train_epochs', default=1, type=int)
    parser.add_argument('--listwise_k', default=20, type=int)
    parser.add_argument('--encoder_output_k', default=-1, type=int)
    args = parser.parse_args()
    args.weight_decay = 0
    args.adam_epsilon = 1e-8
    args.early_stop_callback = False
    args.seed = 0
    args.output_dir = f'./outputs/{args.name}'
    if args.encoder_output_k == -1:
        args.encoder_output_k = args.max_input_length
    try:
        devices = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        devices = ','.join([str(x) for x in list(range(torch.cuda.device_count()))])
    command_str = f"CUDA_VISIBLE_DEVICES={devices} python3 train.py {' '.join(sys.argv[1:])}"
    args.n_gpu = torch.cuda.device_count()
    args.command_str = command_str
    if args.wandb:
        arg2dict = vars(args)
        project_name = 'PUT_YOUR_PROJECT_NAME'
        entity_name = 'PUT_YOUR_ENTITY_NAME'
        wandb_logger = WandbLogger(
            project=project_name, name=args.name, entity=entity_name,
            config=arg2dict
        )
        print(f"Starting wandb with project: {project_name}, name:{args.name}")
    else:
        wandb_logger = None
    if not args.do_train:
        args.n_gpu = 1
        #assert torch.cuda.device_count() == 1
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor='global_step',
        mode='max',
        dirpath=args.output_dir,
        every_n_train_steps=args.eval_steps,
        filename="{epoch:02d}-step{global_step}",
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    if args.lr_scheduler == "constant":
        print(f"@@@ Not Using Learning Rate Scheduler")
    else:
        lr_callback = LearningRateMonitor()
        callbacks.append(lr_callback)

    if args.accelerator == "ddp":
        #plugins = DDPPlugin(find_unused_parameters=False)
        plugins = pl.strategies.DDPStrategy(find_unused_parameters=False)
        print(f"@@@ Using DDP")
    elif args.accelerator == "deepspeed":
        #plugins = pl.strategies.deepspeed.DeepSpeedStrategy(stage=2, load_full_weights=True) #zero_force_ds_cpu_optimizer=False)
        plugins = pl.strategies.deepspeed.DeepSpeedStrategy(stage=2) #,
                #offload_optimizer=True, offload_parameters=True, partition_activations=True) #zero_force_ds_cpu_optimizer=False)
        #plugins = DeepSpeedPlugin(stage=2, offload_optimizer=False, offload_parameters=False, load_full_weights=True)

        print(f"@@@ Using Deepspeed stage2")
    elif args.accelerator == 'dp':
        plugins = 'dp'
        print(f"@@@ Using dp @@@@")
    else:
        import pdb; pdb.set_trace()
        raise NotImplementedError("** accelerator: Choose between (ddp|dp|deepspeed)")

    if args.run_3b:
        args.precision=16
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            accelerator='gpu',
            devices=args.n_gpu,
            #strategy=plugins,
            strategy='deepspeed_stage_2_offload',
            max_epochs=args.num_train_epochs,
            precision=args.precision,
            default_root_dir=args.output_dir,
            logger=wandb_logger,
            val_check_interval=args.eval_steps,
            check_val_every_n_epoch=None,
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )
    else:
        args.precision = 'bf16'
        torch.set_float32_matmul_precision('high')
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            accelerator='gpu',
            devices=args.n_gpu,
            strategy=plugins,
            max_epochs=args.num_train_epochs,
            precision=args.precision,
            default_root_dir=args.output_dir,
            logger=wandb_logger,
            val_check_interval=args.eval_steps,
            check_val_every_n_epoch=None,
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )
    Path(args.output_dir).parent.mkdir(exist_ok=True, parents=True)
    main(args, train_params)


