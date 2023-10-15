import argparse
import os

import numpy as np
import torch
#from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel,OurModel
from tqdm import tqdm
from utils import set_seed, collate_fn
from prepro import read_docred,Rule_Miner,Rule_Miner_our
from evaluation import to_official, official_evaluate
import wandb


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in tqdm(enumerate(train_dataloader)):#这里改了用tqdm
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          'type_id':batch[5]
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                #with amp.scale_loss(loss, optimizer) as scaled_loss:
                #    scaled_loss.backward()
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                wandb.log({"loss": loss.item()}, step=num_steps)
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    test_score, test_output = evaluate(args, model, test_features, tag="test")
                    wandb.log(dev_output, step=num_steps)
                    wandb.log(test_output, step=num_steps)
                    print(dev_output)
                    print(test_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        # pred = report(args, model, test_features) #这边儿先注释掉，不用他生成结果
                        # with open("result.json", "w") as fh:
                        #     json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'type_id':batch[5]
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign = official_evaluate(ans, args.data_dir,tag)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'type_id':batch[5]
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/dwie", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="./dataset/dwie/test", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,#这里原来是-1
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=66,#这里docred是97
                        help="Number of relation types in dataset.")
    #MILR参数
    parser.add_argument("--minC", type=float, default=0.98,
                        help="最小置信度")
    parser.add_argument("--maxL", type=int, default=2,
                        help="规则最大长度")
    parser.add_argument("--Lambda", type=float, default=1e-3,
                        help="超参")
    parser.add_argument("--Eta1", type=float, default=-0.05,
                        help="超参")
    parser.add_argument("--Eta2", type=float, default=-0.1,
                        help="超参")
    parser.add_argument("--T", type=float, default=0.8,
                        help="超参")
    parser.add_argument("--k", type=float, default=0.5,
                        help="超参")
    parser.add_argument("--mode", type=str, default='our',
                        help="普通模式还是加插件")


    #our参数
    parser.add_argument("--minbetaC", type=float, default=0.9,
                        help="最小置信度大于90%的概率")
    parser.add_argument("--minbetaH", type=float, default=0.9,
                        help="最小头部覆盖率大于90%的概率")
    parser.add_argument("--betatheta", type=float, default=0.9,
                        help="最小头部覆盖率大于90%的概率")
    
    args = parser.parse_args()
    wandb.init(project="DocRED")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features,pr = read(train_file, tokenizer, max_seq_length=args.max_seq_length,data_dir=args.data_dir)
    dev_features,_ = read(dev_file, tokenizer, max_seq_length=args.max_seq_length,data_dir=args.data_dir)
    test_features,_ = read(test_file, tokenizer, max_seq_length=args.max_seq_length,data_dir=args.data_dir)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    if args.mode == 'MILR':
        rules,rulestoconf = Rule_Miner(args.data_dir,conf=args.minC,length=args.maxL)
        model = DocREModel(config, model, num_labels=args.num_labels,args=args,rules=rules,rulestoconf=rulestoconf,pr=pr)
    elif args.mode == 'our':
        rulesconf,rulestobetaconf,ruleshead,rulestobetahead = Rule_Miner_our(args.data_dir,minbetaconf=args.minbetaC,minbetahead = args.minbetaH,length=args.maxL,betatheta = args.betatheta)
        model = OurModel(config, model, num_labels=args.num_labels,args=args,rulesconf=rulesconf,rulestobetaconf=rulestobetaconf,ruleshead=ruleshead,rulestobetahead=rulestobetahead,pr=pr)
    else:
        model = DocREModel(config, model, num_labels=args.num_labels,args=args)
    model.to(0)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        #model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(dev_output)
        print(test_output)
        pred = report(args, model, test_features)
        with open("result.json", "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
