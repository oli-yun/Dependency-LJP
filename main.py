# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import random
import os

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from loguru import logger
from parameters import parse
from transformers import AutoTokenizer, Adafactor, MT5ForConditionalGeneration
from train_tools import fit, predict, EarlyStopping
from data_preprocess import preprocess_data, generate_target, generate_decoder_input

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LegalGenerator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        device_map = {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7, 8, 9, 10, 11]
        }
        self.model.parallelize(device_map)

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None, past_key_values=None):
        if decoder_input_ids is None:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
            )
        else:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True
            )

    def generate(self, input_ids, eos_token_id=None):
        return self.model.generate(input_ids, eos_token_id=eos_token_id, max_length=512)


class CAILDataset(Dataset):
    def __init__(self, data, tokenizer, with_interpretation, article_content_dict=None, decoder_input=None):
        self.input = data[0]
        self.target = data[1]
        self.tokenizer = tokenizer
        self.with_interpretation = with_interpretation
        self.article_content_dict = article_content_dict
        self.decoder_input = decoder_input
        self.len = len(self.input)

    def __getitem__(self, item):
        input_data = self.input[item]
        target = self.target[item]

        if input_data.startswith('预测'):
            if input_data.startswith('预测相关法条内容'):
                target = target.split(',')
                for i, article in enumerate(target):
                    target[i] = self.tokenizer.decode(self.article_content_dict[article])
                target = ';'.join(target)
            target = self.tokenizer(target, padding='max_length', max_length=400, truncation=True,
                                    return_tensors='pt')['input_ids'][0]
        elif self.article_content_dict is not None:
            article_content = target[target.find('<extra_id_3>')+12:target.find('<extra_id_4>')].strip()
            target_0 = target[:target.find('<extra_id_3>')+12]
            target_1 = target[target.find(' <extra_id_4>'):]
            article_content = [self.article_content_dict[a] for a in article_content.split(',')]
            target = target_0 + '|'.join(article_content) + target_1
            target = self.tokenizer(target, padding='max_length', max_length=512, truncation=True,
                                    return_tensors='pt')['input_ids'][0]
        elif self.with_interpretation:
            target = self.tokenizer(target, padding='max_length', max_length=400, truncation=True, return_tensors='pt'
                                    )['input_ids'][0]
        else:
            # only generate articles, accusations, penalty
            target = self.tokenizer(target, padding='max_length', max_length=128, truncation=True, return_tensors='pt'
                                    )['input_ids'][0]

        input_data = self.tokenizer(input_data, padding='max_length', max_length=512,
                                    truncation=True, return_tensors='pt')
        # target = self.tokenizer(target, padding='max_length', max_length=128,
        #                         truncation=True, return_tensors='pt')['input_ids'][0]

        if self.decoder_input is not None:
            decoder_input_id = self.tokenizer('<pad>' + self.decoder_input[item], return_tensors='pt', max_length=400,
                                              add_special_tokens=False, padding='max_length', truncation=True)
            given_len = torch.sum(decoder_input_id['attention_mask'])
            return {'input_ids': input_data['input_ids'][0],
                    'attention_mask': input_data['attention_mask'][0],
                    'decoder_input_ids': decoder_input_id['input_ids'][0],
                    'given_len': given_len,
                    'labels': target}
        else:
            return {'input_ids': input_data['input_ids'][0],
                    'attention_mask': input_data['attention_mask'][0],
                    'labels': target}

    def __len__(self):
        return self.len


def main(args):
    set_seed(args.seed)
    start_time = time.strftime("-%Y-%m-%d", time.localtime())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.add(args.log_path + start_time + '.log')

    message = 'Print Setting Parameter: \n'
    for k in list(vars(args).keys()):
        message += '%s: %s\n' % (k, vars(args)[k])
    logger.info(message)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_size = None if args.train_all_data else args.train_data_size
    elements = args.elements.split(',')
    if args.use_article_content:
        with open(args.article_content_path, 'rb') as f:
            article_content_dict = torch.load(f)
    else:
        article_content_dict = None

    ###########
    # 99:<extra_id_0> 98:<extra_id_1> 97:<extra_id_2> 96:<extra_id_3> 95:<extra_id_4> 94:<extra_id_5>
    eos_token_id = None
    ignore_signals = []

    model = LegalGenerator(args.model_name)
    model.to(device)
    model_path = args.model_path + start_time + '.pkl' if args.history_model_path == '' else args.history_model_path
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    early_stopping = EarlyStopping(output_path=model_path, patience=args.early_stopping_patience, verbose=True)

    if args.do_train:
        if args.history_model_path != '':
            state = torch.load(model_path)
            model.load_state_dict(state['net'])
            optimizer.load_state_dict(state['optimizer'])
            del state
        train_dataloader = \
            generate_dataloader(tokenizer, args.train_data_path, elements, args.batch_size, args.supervise,
                                args.with_interpretation, article_content_dict, shuffle=True,
                                use_article_content=args.use_article_content, data_size=data_size)
        if args.supervise:
            val_dataloader = {}
            for element in [ele for ele in elements if ele not in ['view', 'article_content']]:
                val_dataloader[element] = \
                    generate_dataloader(tokenizer, args.valid_data_path, [element], 32, args.supervise,
                                        args.with_interpretation, article_content_dict,
                                        use_article_content=args.use_article_content)
        else:

            val_dataloader = generate_dataloader(tokenizer, args.valid_data_path, elements,
                                                 32, args.supervise,
                                                 args.with_interpretation, article_content_dict,
                                                 use_article_content=args.use_article_content)
        val_targets = generate_target(args.valid_data_path)

        logger.info('Start Training')
        fit(train_dataloader, val_dataloader, val_targets, model, tokenizer, optimizer, early_stopping, args.max_epochs,
            device, args.gradient_accelerator, args.evaluate_per_epoch, args.log_interval, args.evaluate_per_steps,
            eos_token_id, ignore_signals)

    if args.do_test:
        logger.info('Test model with best performance.')
        state = torch.load(model_path)
        model.load_state_dict(state['net'])
        del state
        if args.supervise:
            for element in elements:
                if element not in ['view', 'article_content']:
                    test(model, tokenizer, args, [element], article_content_dict, device, start_time,
                         eos_token_id, ignore_signals)
        else:
            test(model, tokenizer, args, elements, article_content_dict, device, start_time,
                 eos_token_id, ignore_signals)

    logger.success('Done')


def test(model, tokenizer, args, elements, article_content_dict, device, start_time, eos_token_id, ignore_signals):
    logger.info('Elements:' + ','.join(elements))
    test_decoder_input = None
    test_dataloader = \
        generate_dataloader(tokenizer, args.test_data_path, elements, 32, args.supervise, args.with_interpretation,
                            article_content_dict,
                            use_article_content=args.use_article_content,
                            decoder_input=test_decoder_input)
    test_targets = generate_target(args.test_data_path)

    logger.info('Evaluate test data:')
    test_metrics = predict(test_dataloader, test_targets, model, tokenizer, device,
                           args.predict_path + 'test' + start_time + '.txt', eos_token_id, ignore_signals)
    message = ''
    for metric in test_metrics:
        message += '{}: {:.6f}\t'.format(metric, test_metrics[metric])
    logger.info(message)


def generate_dataloader(tokenizer, data_path, elements, batch_size, supervise, with_interpretation, article_content_dict
                        , shuffle=False, use_article_content=False, data_size=None, decoder_input=None):
    data = preprocess_data(tokenizer, data_path, elements=elements, shuffle=shuffle, data_size=data_size,
                           supervise=supervise, use_article_content=use_article_content)
    article_content_dict = None if not use_article_content else article_content_dict
    dataset = CAILDataset(data, tokenizer, with_interpretation,
                          article_content_dict=article_content_dict, decoder_input=decoder_input)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    return dataloader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    arg = parse()
    main(arg)
