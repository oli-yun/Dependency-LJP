import re
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from data_preprocess import get_article_idx, get_accusation_idx, get_penalty_num, get_all_articles


def fit(train_dataloader, dev_dataloader, dev_targets, model, tokenizer, optimizer, early_stopping, epochs,
        device, gradient_accelerator, evaluate_per_epoch, log_interval, evaluate_per_steps,
        eos_token_id, ignore_signals):
    # model.to(device)
    data_len = len(train_dataloader)
    last_gradient_accelerator = data_len % gradient_accelerator
    last_gradient_idx = data_len // gradient_accelerator * gradient_accelerator
    step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        losses = []
        optimizer.zero_grad()
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            data = {k: data[k].to(device) for k in data}
            loss = model(**data).loss
            loss = torch.mean(loss)
            total_loss += loss.item()
            losses.append(loss.item())

            if batch_idx >= last_gradient_idx:
                loss /= last_gradient_accelerator
            else:
                loss /= gradient_accelerator

            loss.backward()

            if (batch_idx + 1) % gradient_accelerator == 0 or batch_idx == data_len - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if not evaluate_per_epoch:
                    step += 1
                    if step % evaluate_per_steps == 0:
                        train_loss = np.mean(losses)
                        logger.info('Step {}. Train set: Average Loss: {:.6f}'.format(step, train_loss))
                        losses = []

                        # val_loss = test_epoch(dev_dataloader, model, device)
                        # message = 'Step {}. Validation set: Average loss: {:.6f}'.format(step, val_loss)
                        val_metrics = {}
                        if isinstance(dev_dataloader, dict):
                            for i, dataloader in enumerate(dev_dataloader):
                                for key, value in predict(dev_dataloader[dataloader], dev_targets, model,
                                                          tokenizer, device, eos_token_id=eos_token_id,
                                                          ignore_signals=ignore_signals).items():
                                    if key.startswith(dataloader):
                                        val_metrics[key] = value
                        else:
                            val_metrics = predict(dev_dataloader, dev_targets, model, tokenizer,
                                                  device, eos_token_id=eos_token_id, ignore_signals=ignore_signals)
                        message = 'Step {}. Validation set: '.format(step)
                        for metric in val_metrics:
                            message += '{}: {:.6f}\t'.format(metric, val_metrics[metric])
                        val_score = val_metrics.get('article_MiF', 0) + val_metrics.get('article_MaF', 0) + \
                                    val_metrics.get('accusation_MiF', 0) + val_metrics.get('accusation_MaF', 0) \
                                    - val_metrics.get('penalty_Score', 0)
                        # flag = early_stopping(val_loss, model, optimizer, step)
                        flag = early_stopping(-val_score, model, optimizer, step)
                        if flag:
                            message += ' *'
                        logger.info(message)
                        if early_stopping.early_stop:
                            break
                        model.train()

            if evaluate_per_epoch:
                if (batch_idx + 1) % log_interval == 0:
                    logger.info('Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                                .format(batch_idx, data_len, 100. * batch_idx / data_len, np.mean(losses)))
                    losses = []

        if evaluate_per_epoch:
            if epoch < 10:
                continue
            total_loss /= data_len
            logger.info('Epoch: {}/{}. Train set: Average loss: {:.6f}'.format(epoch + 1, epochs, total_loss))
            val_metrics = {}
            if isinstance(dev_dataloader, dict):
                for i, dataloader in enumerate(dev_dataloader):
                    for key, value in predict(dev_dataloader[dataloader], dev_targets, model,
                                              tokenizer, device, eos_token_id=eos_token_id,
                                              ignore_signals=ignore_signals).items():
                        if key.startswith(dataloader):
                            val_metrics[key] = value
            else:
                val_metrics = predict(dev_dataloader, dev_targets, model, tokenizer,
                                      device, eos_token_id=eos_token_id, ignore_signals=ignore_signals)
            message = 'Epoch: {}/{}. Validation set: '.format(epoch + 1, epochs)
            for metric in val_metrics:
                message += '{}: {:.6f}\t'.format(metric, val_metrics[metric])
            logger.info(message)
            val_score = val_metrics.get('article_MiF', 0) + val_metrics.get('article_MaF', 0) + \
                        val_metrics.get('accusation_MiF', 0) + val_metrics.get('accusation_MaF', 0) \
                        - val_metrics.get('penalty_Score', 0)
            early_stopping(-val_score, model, optimizer, step)

        if early_stopping.early_stop:
            logger.info('Early stopping!')
            break


def predict(dataloader, targets, model, tokenizer, device, predict_path=None, eos_token_id=None, ignore_signals=[]):
    # model.to(device)
    model.eval()
    with torch.no_grad():
        res = []
        for data in tqdm(dataloader):
            if 'decoder_input_ids' in data:
                given_len = data['given_len'].to(device)
                given_decoder_input_id = data['decoder_input_ids'].to(device)
                data = {k: data[k].to(device) for k in data if k not in ['labels', 'given_len', 'decoder_input_ids']}
                cur_len = 0
                data['decoder_input_ids'] = torch.zeros((data['input_ids'].size(0), 1)).type_as(data['input_ids'])
                output_text = torch.zeros((data['input_ids'].size(0), 1)).type_as(data['input_ids'])
                key_values = None
                stop_cnt = 0
                while cur_len < 399:
                    cur_len += 1
                    outputs = model(**data, past_key_values=key_values)
                    # outputs = model(**data)
                    pred = torch.argmax(outputs.logits, dim=-1)[:, -1]
                    if pred.sum() == data['input_ids'].size(0):
                        break
                    this_pos = torch.where(torch.ones((data['input_ids'].size(0)),
                                                      dtype=torch.int, device=device) * cur_len < given_len,
                                           given_decoder_input_id[:, cur_len], pred)
                    stop_cnt += torch.sum(torch.where(
                        pred == 250097, torch.ones(pred.size()).type_as(pred),
                        torch.zeros(pred.size()).type_as(pred))).item()
                    # data['decoder_input_ids'] = torch.cat([data['decoder_input_ids'], this_pos.unsqueeze(dim=1)], dim=-1)
                    output_text = torch.cat([output_text, this_pos.unsqueeze(dim=1)], dim=-1)
                    data['decoder_input_ids'] = this_pos.unsqueeze(dim=1)
                    key_values = outputs.past_key_values
                    if stop_cnt == data['input_ids'].size(0):
                        break
                # res.extend(get_labels(data['decoder_input_ids'], tokenizer, predict_path))
                res.extend(get_labels(output_text, tokenizer, predict_path))
            else:
                pred = model.generate(data['input_ids'].to(device), eos_token_id=eos_token_id).detach().cpu().numpy()
                res.extend(get_labels(pred, tokenizer, predict_path, ignore_signals=ignore_signals))
    metrics = get_metrics(res, targets)
    return metrics


def test_epoch(dataloader, model, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in tqdm(dataloader):
            data = {k: data[k].to(device) for k in data}
            loss = model(**data).loss
            total_loss += loss.item()
    total_loss /= len(dataloader)
    return total_loss


def match_content(content, res):
    if re.match(r'第.*?条', content) is not None:
        for article in content.split(','):
            idx = get_article_idx(article)
            if idx is not None:
                res['articles'][idx] = 1
    elif re.match(r'.*罪$', content) is not None:
        for accu in content.split(','):
            idx = get_accusation_idx(accu)
            if idx is not None:
                res['accusation'][idx] = 1
    elif re.match(r'有期徒刑', content) is not None or content == '死刑' or content == '无期徒刑':
        idx = get_penalty_num(content)
        if idx is not None:
            res['imprisonment'][0] = idx
    return res


def get_labels(preds, tokenizer, predict_path, ignore_signals=[]):
    results = []
    for pred in preds:
        res = {"accusation": [0] * 200, "articles": [0] * 183, "imprisonment": [0]}
        text = tokenizer.decode(pred, skip_special_tokens=True)
        if predict_path is not None:
            with open(predict_path, 'a', encoding='utf-8') as f:
                f.write(text + '\n')

        if text.startswith('<extra'):
            signal_pair = [('<extra_id_0>', '<extra_id_1>'), ('<extra_id_1>', '<extra_id_2>'),
                           ('<extra_id_2>', '<extra_id_3>'), ('<extra_id_3>', '<extra_id_4>'),
                           ('<extra_id_4>', '<extra_id_5>')]
            for pair in signal_pair:
                if pair in ignore_signals:
                    continue
                ret = re.search(r'{}.*?{}'.format(pair[0], pair[1]), text)
                if ret is not None:
                    content = ret.group().replace(pair[0], '').replace(pair[1], '').strip()
                    res = match_content(content, res)
        else:
            res = match_content(text, res)
        results.append(res)
    return results


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, output_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        # self.val_score = 0
        self.delta = delta
        self.output_path = output_path

    def __call__(self, val_score, model, optimizer, epoch):

        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(val_score, model, optimizer, epoch)
            return True
        elif val_score >= self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = val_score
            self.save_checkpoint(val_score, model, optimizer, epoch)
            self.counter = 0
            return True

    def save_checkpoint(self, val_score, model, optimizer, epoch):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            logger.info(f'Validation score decreased ({self.val_loss_min:.6f} --> {val_score:.6f}).  Saving model ...')
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, self.output_path)
        self.val_loss_min = val_score


def get_metrics(preds, targets):
    result = [None, None, None]
    pred = {'articles': [], 'accusation': [], 'imprisonment': []}
    target = {'articles': [], 'accusation': [], 'imprisonment': []}

    for p, t in zip(preds, targets):
        pred['articles'].append(torch.tensor([p['articles']]))
        pred['accusation'].append(torch.tensor([p['accusation']]))
        pred['imprisonment'].append(torch.tensor(p['imprisonment']))
        target['articles'].append(torch.tensor([t['articles']]))
        target['accusation'].append(torch.tensor([t['accusation']]))
        target['imprisonment'].append(torch.tensor(t['imprisonment']))
    gen_new_result(pred, target, result)

    metrics = get_score(result)
    return metrics


def gen_new_result(label, truth, result):
    result[0] = multi_label_accuracy(
        torch.cat(label['articles'], dim=0), torch.cat(truth['articles'], dim=0), result[0]
    )
    result[1] = multi_label_accuracy(
        torch.cat(label['accusation'], dim=0), torch.cat(truth['accusation'], dim=0), result[1]
    )
    result[2] = log_distance_accuracy_function(
        torch.cat(label['imprisonment'], dim=0), torch.cat(truth['imprisonment'], dim=0), result[2]
    )


def multi_label_accuracy(outputs, label, result=None):
    if len(label[0]) != len(outputs[0]):
        raise ValueError('Input dimensions of labels and outputs must match.')

    if len(outputs.size()) > 2:
        outputs = outputs.view(outputs.size()[0], -1, 2)
        outputs = torch.nn.Softmax(dim=2)(outputs)
        outputs = outputs[:, :, 1]

    outputs = outputs.data
    labels = label.data

    if result is None:
        result = []

    total = 0
    nr_classes = outputs.size(1)

    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

    for i in range(nr_classes):
        outputs1 = (outputs[:, i] >= 0.5).long()
        labels1 = (labels[:, i].float() >= 0.5).long()
        total += int((labels1 * outputs1).sum())
        total += int(((1 - labels1) * (1 - outputs1)).sum())

        if result is None:
            continue

        # if len(result) < i:
        #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        result[i]["TP"] += int((labels1 * outputs1).sum())
        result[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        result[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        result[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())

    return result


def log_distance_accuracy_function(outputs, label, result=None):
    if result is None:
        result = [0, 0]

    result[0] += label.size()[0]
    result[1] += float(
        torch.sum(torch.log(torch.abs(torch.clamp(outputs.float(), 0, 450) - torch.clamp(label.float(), 0, 450)) + 1)))

    return result


def get_score(result):
    Art_MiP, Art_MiR, Art_MiF, Art_MaP, Art_MaR, Art_MaF = gen_score(result[0])
    Acc_MiP, Acc_MiR, Acc_MiF, Acc_MaP, Acc_MaR, Acc_MaF = gen_score(result[1])
    Pen_score = result[2]
    metrics = {'article_MiP': Art_MiP, 'article_MiR': Art_MiR, 'article_MiF': Art_MiF,
               'article_MaP': Art_MaP, 'article_MaR': Art_MaR, 'article_MaF': Art_MaF,
               'accusation_MiP': Acc_MiP, 'accusation_MiR': Acc_MiR, 'accusation_MiF': Acc_MiF,
               'accusation_MaP': Acc_MaP, 'accusation_MaR': Acc_MaR, 'accusation_MaF': Acc_MaF,
               'penalty_Score': Pen_score[1] / Pen_score[0]}
    return metrics


def gen_score(res):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_prf(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return round(micro_precision, 3), round(micro_recall, 3), \
           round(micro_f1, 3), round(macro_precision, 3), round(macro_recall, 3), round(macro_f1, 3)


def get_prf(res):
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1
