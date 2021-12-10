# -*- coding:utf-8 -*-
import re
import json
import torch
import random
from tqdm import tqdm


def preprocess_data(tokenizer, data_path, elements, data_size=None, shuffle=False,
                    supervise=False, use_article_content=False):
    """
    supervise:
        input: [Task Prefix] [fact]
        output: [articles or accusations or penalty]
    un-supervise:
        input: [fact] 综上，依据《中华人民共和国刑法》<extra_id_0>的规定，被告人行为构成<extra_id_1>，应判处<extra_id_2>。本院认为：<extra_id_3>。
        output: <extra_id_0>第x条,第x条<extra_id_1>x罪,x罪<extra_id_2>有期徒刑x个月/死刑、无期徒刑<extra_id_3>[court_view]<extra_id_4>
    :param use_article_content:
    :param tokenizer:
    :param data_path: json file path
    :param elements: subset of ['article', 'accusation', 'view', 'penalty']
    :param data_size:
    :param shuffle:
    :param supervise:
    :return:
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        if shuffle:
            random.seed(1024)
            random.shuffle(all_lines)
        if data_size is not None:
            all_lines = all_lines[:data_size]

    input_datas, targets = [], []
    if supervise:
        for line in all_lines:
            input_data, target = prepare_supervised_data(line, elements)
            input_datas += input_data
            targets += target
    else:
        for line in all_lines:
            input_data, target = prepare_unsupervised_data(tokenizer, line, elements, use_article_content)
            input_datas += input_data
            targets += target

    return input_datas, targets


def generate_target(data_path):
    results = []
    with open(data_path, 'r', encoding='utf-8') as f_r:
        for line in tqdm(f_r.readlines()):
            res = {"accusation": [0] * 200, "articles": [0] * 183, "imprisonment": []}
            data = json.loads(line)
            relevant_articles = data['meta']['relevant_articles']
            for article in relevant_articles:
                article = '第' + str(article) + '条'
                res['articles'][get_article_idx(article)] = 1
            accusation = data['meta']['accusation']
            for acc in accusation:
                acc = acc.replace('[', '').replace(']', '') + '罪'
                res['accusation'][get_accusation_idx(acc)] = 1
            term_of_imprisonment = data['meta']['term_of_imprisonment']
            if term_of_imprisonment['death_penalty']:
                num = -2
            elif term_of_imprisonment['life_imprisonment']:
                num = -1
            else:
                num = term_of_imprisonment['imprisonment']

            res['imprisonment'] = [num]
            results.append(res)
    return results


def get_elements_from_line(line):
    data = json.loads(line)
    fact = data['fact'].replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
    view = re.sub(r'本院认为.', '', data.get('interpretation', '').strip())
    view = view.replace('\n', '').replace('\t', '').replace('\r', '') \
        .replace('&times；', 'x').replace('&divide；', '÷').replace('&amp；', '&').replace('&yen；', '￥')
    view = re.sub(r'&ldquo；|&rdquo；|&middot；|&quot；|&plusmn；|&hellip；|&lsquo；|&rsquo；|&permil；|&mdash；|\|', '', view)
    articles = ','.join(['第' + str(num) + '条' for num in sorted(data['meta']['relevant_articles'])])
    accusations = ','.join([accu.strip() + '罪' for accu in set(data['meta']['accusation'])])
    # article_contents = articles.split(',')[-1]
    article_contents = articles
    if data['meta']['term_of_imprisonment']['death_penalty']:
        penalty = '死刑'
    elif data['meta']['term_of_imprisonment']['life_imprisonment']:
        penalty = '无期徒刑'
    else:
        penalty_length = data['meta']['term_of_imprisonment']['imprisonment']
        penalty = '有期徒刑' + str(penalty_length) + '个月'
    return fact, articles, accusations, penalty, view, article_contents


def prepare_supervised_data(line, elements):
    fact, articles, accusations, penalty, view, article_content = get_elements_from_line(line)
    input_datas, targets = [], []
    if 'accusation' in elements:
        input_datas.append('预测案由：' + fact)
        targets.append(accusations)
    if 'penalty' in elements:
        input_datas.append('预测刑期：' + fact)
        targets.append(penalty)
    if 'view' in elements:
        input_datas.append('预测法庭意见：' + fact)
        targets.append(view)
    if 'article' in elements:
        input_datas.append('预测相关法条条目：' + fact)
        targets.append(articles)
    if 'article_content' in elements:
        input_datas.append('预测相关法条内容：' + fact)
        targets.append(articles)
    return input_datas, targets


def prepare_unsupervised_data(tokenizer, line, elements, use_article_content):
    facts, targets = [], []
    fact, articles, accusations, penalty, view, article_content = get_elements_from_line(line)
    special_token_maps = {i: ' <extra_id_' + str(i) + '>' for i in range(6)}
    add_text = ''
    target = ''
    token_counter = 0

    if 'article' in elements:
        add_text += '综上，依据《中华人民共和国刑法》' + special_token_maps[token_counter] + '的规定，'
        target += special_token_maps[token_counter] + articles
        token_counter += 1
    if 'accusation' in elements:
        add_text += '被告人行为构成' + special_token_maps[token_counter] + '，'
        target += special_token_maps[token_counter] + accusations
        token_counter += 1
    # if 'accusation' in elements:
    #     add_text += '综上，被告人行为构成' + special_token_maps[token_counter] + '，'
    #     target += special_token_maps[token_counter] + accusations
    #     token_counter += 1
    # if 'article' in elements:
    #     add_text += '依据《中华人民共和国刑法》' + special_token_maps[token_counter] + '的规定，'
    #     target += special_token_maps[token_counter] + articles
    #     token_counter += 1
    if 'penalty' in elements:
        add_text += '应判处' + special_token_maps[token_counter] + '。'
        target += special_token_maps[token_counter] + penalty
        token_counter += 1
    if use_article_content:
        add_text += '对应的法条内容有：' + special_token_maps[token_counter] + '。'
        target += special_token_maps[token_counter] + article_content
        token_counter += 1
    if 'view' in elements:
        add_text += '本院认为：' + special_token_maps[token_counter] + '。'
        # target += special_token_maps[token_counter] + view[:-1]
        view = view[:-1]
        # input_ids = tokenizer.encode(view, truncation=True, max_length=256)
        # if len(input_ids) == 256:
        #     view = tokenizer.decode(input_ids[:255])
        target += special_token_maps[token_counter] + view
        token_counter += 1

    # cut input length
    input_data = add_text + fact
    input_ids = tokenizer.encode(input_data, truncation=True, max_length=512)
    if len(input_ids) == 512:
        idx = 250100 - token_counter
        fact = tokenizer.decode(input_ids[input_ids.index(idx) + 2:-1]) + '…'
    facts.append(fact + add_text)
    targets.append(target.strip() + special_token_maps[token_counter])

    # if 'view' in elements:
    #     facts.append(fact + '本院认为：<extra_id_0>。')
    #     targets.append('<extra_id_0>' + view[:-1] + ' <extra_id_1>')

    # if use_article_content:
    #     # modify the way adding article content
    #     facts.append(fact + '涉及的法条内容有：<extra_id_0>。')
    #     targets.append(article_content)
    return facts, targets


def generate_decoder_input(data_path, article_content_dict, tokenizer, test_path):
    decoder_inputs = []
    with open(data_path, 'r', encoding='utf-8') as f, open(test_path, 'r', encoding='utf-8') as f_2:
        for line, target in zip(f.readlines(), f_2.readlines()):
            fact, articles, accusations, penalty, view, article_contents = get_elements_from_line(target)
            article_contents = article_contents.split(',')
            for i, article in enumerate(article_contents):
                article_contents[i] = tokenizer.decode(article_content_dict[article])
            article_contents = ';'.join(article_contents)
            decoder_input = '<extra_id_0>' + articles + ' <extra_id_1>'
            # decoder_input = '<extra_id_0>' + articles + ' <extra_id_1>' + accusations + ' <extra_id_2>'
            # decoder_input = '<extra_id_0>' + articles + ' <extra_id_1>' + article_contents + ' <extra_id_2>' + \
            #                 accusations + ' <extra_id_3>'
            decoder_inputs.append(decoder_input)

    return decoder_inputs


def get_article_idx(article):
    article2idx = get_all_articles()
    if article in article2idx:
        return article2idx[article]
    else:
        return None


def get_all_articles():
    return {'第347条': 0, '第264条': 1, '第383条': 2, '第234条': 3, '第133条': 4, '第196条': 5, '第280条': 6, '第345条': 7,
            '第266条': 8, '第397条': 9, '第263条': 10, '第205条': 11, '第385条': 12, '第303条': 13, '第386条': 14, '第356条': 15,
            '第114条': 16, '第224条': 17, '第293条': 18, '第276条': 19, '第238条': 20, '第357条': 21, '第277条': 22,
            '第354条': 23, '第382条': 24, '第390条': 25, '第236条': 26, '第115条': 27, '第128条': 28, '第163条': 29,
            '第267条': 30, '第134条': 31, '第125条': 32, '第271条': 33, '第232条': 34, '第389条': 35, '第342条': 36,
            '第292条': 37, '第312条': 38, '第336条': 39, '第176条': 40, '第272条': 41, '第269条': 42, '第141条': 43,
            '第348条': 44, '第274条': 45, '第275条': 46, '第225条': 47, '第233条': 48, '第384条': 49, '第359条': 50,
            '第363条': 51, '第351条': 52, '第175条': 53, '第313条': 54, '第214条': 55, '第144条': 56, '第143条': 57,
            '第338条': 58, '第310条': 59, '第237条': 60, '第344条': 61, '第341条': 62, '第213条': 63, '第245条': 64,
            '第177条': 65, '第279条': 66, '第393条': 67, '第140条': 68, '第153条': 69, '第235条': 70, '第364条': 71,
            '第340条': 72, '第118条': 73, '第343条': 74, '第358条': 75, '第192条': 76, '第124条': 77, '第239条': 78,
            '第240条': 79, '第258条': 80, '第210条': 81, '第307条': 82, '第388条': 83, '第164条': 84, '第253条': 85,
            '第346条': 86, '第172条': 87, '第314条': 88, '第288条': 89, '第217条': 90, '第135条': 91, '第290条': 92,
            '第209条': 93, '第228条': 94, '第150条': 95, '第328条': 96, '第198条': 97, '第231条': 98, '第291条': 99,
            '第151条': 100, '第186条': 101, '第226条': 102, '第243条': 103, '第294条': 104, '第194条': 105, '第193条': 106,
            '第372条': 107, '第392条': 108, '第350条': 109, '第171条': 110, '第353条': 111, '第316条': 112, '第387条': 113,
            '第201条': 114, '第223条': 115, '第220条': 116, '第399条': 117, '第396条': 118, '第417条': 119, '第158条': 120,
            '第130条': 121, '第270条': 122, '第391条': 123, '第305条': 124, '第262条': 125, '第349条': 126, '第246条': 127,
            '第155条': 128, '第261条': 129, '第162条': 130, '第286条': 131, '第413条': 132, '第283条': 133, '第215条': 134,
            '第132条': 135, '第375条': 136, '第156条': 137, '第333条': 138, '第315条': 139, '第361条': 140, '第241条': 141,
            '第302条': 142, '第367条': 143, '第159条': 144, '第185条': 145, '第360条': 146, '第395条': 147, '第152条': 148,
            '第168条': 149, '第227条': 150, '第127条': 151, '第260条': 152, '第211条': 153, '第149条': 154, '第282条': 155,
            '第285条': 156, '第136条': 157, '第117条': 158, '第281条': 159, '第295条': 160, '第147条': 161, '第119条': 162,
            '第199条': 163, '第244条': 164, '第418条': 165, '第273条': 166, '第184条': 167, '第402条': 168, '第200条': 169,
            '第369条': 170, '第212条': 171, '第170条': 172, '第248条': 173, '第352条': 174, '第308条': 175, '第161条': 176,
            '第122条': 177, '第191条': 178, '第404条': 179, '第116条': 180, '第326条': 181, '第268条': 182}


def get_accusation_idx(accu):
    accusation2idx = {'盗窃罪': 0, '走私、贩卖、运输、制造毒品罪': 1, '故意伤害罪': 2, '抢劫罪': 3, '诈骗罪': 4, '受贿罪': 5, '寻衅滋事罪': 6, '危险驾驶罪': 7,
                      '组织、强迫、引诱、容留、介绍卖淫罪': 8, '制造、贩卖、传播淫秽物品罪': 9, '容留他人吸毒罪': 10, '交通肇事罪': 11, '贪污罪': 12,
                      '非法持有、私藏枪支、弹药罪': 13, '故意杀人罪': 14, '开设赌场罪': 15, '非法持有毒品罪': 16, '职务侵占罪': 17, '强奸罪': 18,
                      '伪造、变造、买卖国家机关公文、证件、印章罪': 19, '敲诈勒索罪': 20, '掩饰、隐瞒犯罪所得、犯罪所得收益罪': 21, '信用卡诈骗罪': 22, '抢夺罪': 23,
                      '非法占用农用地罪': 24, '赌博罪': 25, '合同诈骗罪': 26, '故意毁坏财物罪': 27, '窝藏、包庇罪': 28,
                      '虚开增值税专用发票、用于骗取出口退税、抵扣税款发票罪': 29, '非法经营罪': 30, '非法吸收公众存款罪': 31, '滥伐林木罪': 32, '妨害公务罪': 33,
                      '行贿罪': 34, '非法拘禁罪': 35, '挪用资金罪': 36, '引诱、容留、介绍卖淫罪': 37, '假冒注册商标罪': 38, '挪用公款罪': 39, '过失致人死亡罪': 40,
                      '盗伐林木罪': 41, '生产、销售假药罪': 42, '制作、复制、出版、贩卖、传播淫秽物品牟利罪': 43, '重大责任事故罪': 44, '拒不执行判决、裁定罪': 45,
                      '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物罪': 46, '放火罪': 47, '生产、销售有毒、有害食品罪': 48, '滥用职权罪': 49,
                      '生产、销售不符合安全标准的食品罪': 50, '污染环境罪': 51, '销售假冒注册商标的商品罪': 52, '非国家工作人员受贿罪': 53, '失火罪': 54,
                      '拒不支付劳动报酬罪': 55, '玩忽职守罪': 56, '骗取贷款、票据承兑、金融票证罪': 57, '伪造公司、企业、事业单位、人民团体印章罪': 58, '聚众斗殴罪': 59,
                      '非法种植毒品原植物罪': 60, '非法行医罪': 61, '非法采伐、毁坏国家重点保护植物罪': 62, '非法狩猎罪': 63, '非法侵入住宅罪': 64,
                      '组织、领导传销活动罪': 65, '传播淫秽物品罪': 66, '单位行贿罪': 67, '生产、销售伪劣产品罪': 68, '招摇撞骗罪': 69, '妨害信用卡管理罪': 70,
                      '过失致人重伤罪': 71, '以危险方法危害公共安全罪': 72, '猥亵儿童罪': 73, '走私普通货物、物品罪': 74, '非法采矿罪': 75, '非法捕捞水产品罪': 76,
                      '集资诈骗罪': 77, '绑架罪': 78, '破坏生产经营罪': 79, '虚开发票罪': 80, '拐卖妇女、儿童罪': 81, '破坏广播电视设施、公用电信设施罪': 82,
                      '重婚罪': 83, '破坏电力设备罪': 84, '持有伪造的发票罪': 85, '强制猥亵、侮辱妇女罪': 86, '伪造、变造居民身份证罪': 87,
                      '非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品罪': 88, '侵犯著作权罪': 89, '非法猎捕、杀害珍贵、濒危野生动物罪': 90,
                      '非法处置查封、扣押、冻结的财产罪': 91, '扰乱无线电通讯管理秩序罪': 92, '重大劳动安全事故罪': 93, '投放危险物质罪': 94, '爆炸罪': 95,
                      '持有、使用假币罪': 96, '非法转让、倒卖土地使用权罪': 97, '盗掘古文化遗址、古墓葬罪': 98, '保险诈骗罪': 99, '组织卖淫罪': 100,
                      '聚众扰乱社会秩序罪': 101, '协助组织卖淫罪': 102, '对非国家工作人员行贿罪': 103, '妨害作证罪': 104, '破坏易燃易爆设备罪': 105,
                      '强迫交易罪': 106, '贷款诈骗罪': 107, '违法发放贷款罪': 108, '诬告陷害罪': 109, '脱逃罪': 110, '非法获取公民个人信息罪': 111,
                      '票据诈骗罪': 112, '冒充军人招摇撞骗罪': 113, '伪造、变造金融票证罪': 114, '出售、购买、运输假币罪': 115, '非法收购、运输盗伐、滥伐的林木罪': 116,
                      '帮助毁灭、伪造证据罪': 117, '非法买卖制毒物品罪': 118, '介绍贿赂罪': 119, '单位受贿罪': 120, '串通投标罪': 121,
                      '聚众扰乱公共场所秩序、交通秩序罪': 122, '逃税罪': 123, '非法进行节育手术罪': 124, '帮助犯罪分子逃避处罚罪': 125,
                      '组织、领导、参加黑社会性质组织罪': 126, '私分国有资产罪': 127, '虚报注册资本罪': 128, '伪证罪': 129, '过失以危险方法危害公共安全罪': 130,
                      '拐骗儿童罪': 131, '编造、故意传播虚假恐怖信息罪': 132, '对单位行贿罪': 133, '引诱、教唆、欺骗他人吸毒罪': 134, '徇私枉法罪': 135,
                      '遗弃罪': 136, '非法出售发票罪': 137, '隐匿、故意销毁会计凭证、会计帐簿、财务会计报告罪': 138, '侵占罪': 139, '窝藏、转移、隐瞒毒品、毒赃罪': 140,
                      '动植物检疫徇私舞弊罪': 141, '走私珍贵动物、珍贵动物制品罪': 142, '伪造、变造、买卖武装部队公文、证件、印章罪': 143,
                      '非法携带枪支、弹药、管制刀具、危险物品危及公共安全罪': 144, '走私国家禁止进出口的货物、物品罪': 145, '非法制造、出售非法制造的发票罪': 146,
                      '利用影响力受贿罪': 147, '破坏计算机信息系统罪': 148, '破坏监管秩序罪': 149, '强迫卖淫罪': 150, '窝藏、转移、收购、销售赃物罪': 151,
                      '非法制造、销售非法制造的注册商标标识罪': 152, '非法生产、销售间谍专用器材罪': 153, '侮辱罪': 154, '传播性病罪': 155, '走私武器、弹药罪': 156,
                      '盗窃、抢夺枪支、弹药、爆炸物、危险物质罪': 157, '窃取、收买、非法提供信用卡信息罪': 158, '盗窃、抢夺枪支、弹药、爆炸物罪': 159, '巨额财产来源不明罪': 160,
                      '非法组织卖血罪': 161, '非法制造、买卖、运输、储存危险物质罪': 162, '破坏交通设施罪': 163, '危险物品肇事罪': 164, '聚众冲击国家机关罪': 165,
                      '盗窃、侮辱尸体罪': 166, '传授犯罪方法罪': 167, '收买被拐卖的妇女、儿童罪': 168, '招收公务员、学生徇私舞弊罪': 169, '非法生产、买卖警用装备罪': 170,
                      '生产、销售伪劣农药、兽药、化肥、种子罪': 171, '过失损坏广播电视设施、公用电信设施罪': 172, '虐待罪': 173, '走私废物罪': 174, '过失投放危险物质罪': 175,
                      '非法获取国家秘密罪': 176, '诽谤罪': 177, '徇私舞弊不移交刑事案件罪': 178, '挪用特定款物罪': 179, '包庇毒品犯罪分子罪': 180, '伪造货币罪': 181,
                      '伪造、倒卖伪造的有价票证罪': 182, '强迫劳动罪': 183, '打击报复证人罪': 184, '强迫他人吸毒罪': 185,
                      '非法买卖、运输、携带、持有毒品原植物种子、幼苗罪': 186, '金融凭证诈骗罪': 187, '非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品罪': 188,
                      '过失损坏武器装备、军事设施、军事通信罪': 189, '提供侵入、非法控制计算机信息系统程序、工具罪': 190, '劫持船只、汽车罪': 191, '洗钱罪': 192,
                      '徇私舞弊不征、少征税款罪': 193, '聚众哄抢罪': 194, '倒卖车票、船票罪': 195, '破坏交通工具罪': 196, '高利转贷罪': 197, '倒卖文物罪': 198,
                      '虐待被监管人罪': 199}
    if accu in accusation2idx:
        return accusation2idx[accu]
    else:
        return None


def get_penalty_num(penalty):
    if penalty == '死刑':
        return 400
    elif penalty == '无期徒刑':
        return 350
    else:
        ret = re.search(r'\d+', penalty)
        return int(re.search(r'\d+', penalty).group(0)) if ret is not None else None
