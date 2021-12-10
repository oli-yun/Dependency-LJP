import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=919)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accelerator', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--evaluate_per_steps', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--early_stopping_patience', type=int, default=3)

    parser.add_argument('--train_data_path', type=str, default='./data/data_train.json')
    parser.add_argument('--valid_data_path', type=str, default='./data/data_valid.json')
    parser.add_argument('--test_data_path', type=str, default='./data/data_test.json')
    parser.add_argument('--article_content_path', type=str, default='./data/article_content_dict.pkl')
    parser.add_argument('--elements', type=str, default='article')
    parser.add_argument('--log_path', type=str, default='./log/accusation_generate_with_content')
    parser.add_argument('--model_path', type=str, default='./checkpoints/accusation_generate_with_content')
    parser.add_argument('--predict_path', type=str, default='./predicts/accusation_generate_with_content_')
    parser.add_argument('--model_name', type=str, default='../mt5-base')
    parser.add_argument('--train_data_size', type=int, default=10000)

    parser.add_argument('--with_interpretation', action='store_true')
    parser.add_argument('--train_all_data', action='store_true')
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--evaluate_per_epoch', action='store_true')
    parser.add_argument('--use_article_content', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--history_model_path', default='')

    args = parser.parse_args()
    return args
