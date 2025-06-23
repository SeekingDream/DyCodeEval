import datasets
import os
import pandas as pd

def get_rl_data(split, dataset, dataset_seed):
    base_path = ''
    assert dataset in ['rl-agnews', 'rl-cr', 'rl-mr', 'rl-sst-2', 
                        'rl-sst-5', 'rl-yelp-2', 'rl-yelp-5']
    num_shots = 16
    seed_dict = {0:'16-100', 1:'16-13', 2:'16-21', 3:'16-42', 4:'16-87'}
    seed_path = seed_dict[dataset_seed]
    split_data_name = dataset.split('rl-')[1]
    filepath = f'{num_shots}-shot/{split_data_name}/{seed_path}/{split}.tsv'
    full_filepath = os.path.join(base_path, filepath)
    df = pd.read_csv(full_filepath, sep='\t')
    if 'text' in df:
        source_texts = df.text.tolist()
    else: 
        source_texts = df.sentence.tolist()
    class_labels = df.label.tolist()
    data = {}
    data['text'] = source_texts
    data['label'] = class_labels
    data = datasets.Dataset.from_dict(data)
    return data

class PromptedClassificationDataset:
    def __init__(self, args):
        self.args = args
        self.glue_list = ['sst2', 'rte', 'mrpc', 'qqp', 'mnli', 'qnli']
        self.superglue_list = ['cb', 'copa', 'boolq', 'wic', 'wsc']
        self.nli_3_list = ['mnli', 'xnli', 'anli', 'cb', 'snli']
        if 'xnli' in args['dataset_name']:
            split = self.args['dataset_name'].split('_')[1]
            self.dataset = datasets.load_dataset('xnli', split)
        elif args['dataset_name'] in self.glue_list:
            self.dataset = datasets.load_dataset('glue', args['dataset_name'])
        elif 'anli' in args['dataset_name']:
            self.dataset = datasets.load_dataset('anli')
        elif args['dataset_name'] in self.superglue_list:
            self.dataset = datasets.load_dataset('super_glue', args['dataset_name'])
        elif 'rl' in args['dataset_name']:
            pass
        else:
            self.dataset = datasets.load_dataset(args['dataset_name'])
    def get_few_shot_dataset(self, shots: int) -> tuple:
        """
        Retrieves a few-shot dataset by selecting a specified number of instances per class from the given dataset.
        
        Args:
            dataset (dict): A dictionary containing the dataset split into "train", "validation", and "test" subsets.
            shots (int): The number of instances to select per class for the few-shot dataset.
        
        Returns:
            tuple: The few-shot training dataset, the original validation dataset, and the original test dataset.
        """
        
        if self.args['dataset_name'] == 'mnli':
            train_dataset = self.dataset['train']
            val_dataset = self.dataset['validation_matched']
            test_dataset = self.dataset['test_matched']
        elif self.args['dataset_name'] == 'yelp_polarity' or self.args['dataset_name'] == 'ag_news' or self.args['dataset_name'] == 'SetFit/CR' or self.args['dataset_name'] == 'yelp_review_full':
            train_dataset = self.dataset['train']
            val_dataset = self.dataset['train']
            test_dataset = self.dataset['test']
        elif 'rl' in self.args['dataset_name']:
            train_dataset = get_rl_data('train', self.args['dataset_name'], self.args['seed'])
            val_dataset = get_rl_data('dev', self.args['dataset_name'], self.args['seed'])
            test_dataset = get_rl_data('test', self.args['dataset_name'], self.args['seed'])
            train_dataset = [x for x in train_dataset]
            val_dataset = [x for x in val_dataset]
            return train_dataset, val_dataset, test_dataset
        elif self.args['dataset_name'] == 'snli':
            train_dataset = [x for x in self.dataset['train'] if x['label'] != -1]
            val_dataset = [x for x in self.dataset['validation'] if x['label'] != -1]
            test_dataset = [x for x in self.dataset['test'] if x['label'] != -1]
        else:
            train_dataset = self.dataset['train']
            val_dataset = self.dataset['validation']
            test_dataset = self.dataset['test']

        train_0 = [x for x in train_dataset if x['label'] == 0][:shots]
        train_1 = [x for x in train_dataset if x['label'] == 1][:shots]
        train_2 = [x for x in train_dataset if x['label'] == 2][:shots]
        train_3 = [x for x in train_dataset if x['label'] == 3][:shots]
        train_4 = [x for x in train_dataset if x['label'] == 4][:shots]
        train_dataset = train_0 + train_1 + train_2 + train_3 + train_4
        if self.args['dataset_name'] in self.glue_list or self.args['dataset_name'] in self.superglue_list:
            val_0 = [x for x in train_dataset if x['label'] == 0][-shots:]
            val_1 = [x for x in train_dataset if x['label'] == 1][-shots:]
            val_2 = [x for x in train_dataset if x['label'] == 2][-shots:]
            new_val_dataset = val_0 + val_1 + val_2
            test_dataset = val_dataset
            print('train_dataset', train_dataset)
            return train_dataset, new_val_dataset, test_dataset
        elif self.args['dataset_name'] == 'ag_news' or self.args['dataset_name'] == 'yele_review_full':
            val_0 = [x for x in train_dataset if x['label'] == 0][-shots:]
            val_1 = [x for x in train_dataset if x['label'] == 1][-shots:]
            val_2 = [x for x in train_dataset if x['label'] == 2][-shots:]
            val_3 = [x for x in train_dataset if x['label'] == 3][-shots:]
            val_4 = [x for x in train_dataset if x['label'] == 4][-shots:]
            new_val_dataset = val_0 + val_1 + val_2 + val_3 + val_4
            test_dataset = val_dataset
            print('train_dataset', train_dataset)
            return train_dataset, new_val_dataset, test_dataset
      
        val_0 = [x for x in val_dataset if x['label'] == 0][:shots]
        val_1 = [x for x in val_dataset if x['label'] == 1][:shots]
        val_2 = [x for x in val_dataset if x['label'] == 2][:shots]
        val_dataset = val_0 + val_1 + val_2
        print('train_dataset', train_dataset)
        return train_dataset, val_dataset, test_dataset

    def get_verbalizer(self) -> list:
        if 'xnli' in self.args['dataset_name'] or self.args['dataset_name'] == 'mnli' or 'anli' in self.args['dataset_name'] or 'americas_nli' in self.args['dataset_name'] or self.args['dataset_name'] == 'snli':
            verbalizer_predefined = ['yes', 'maybe', 'no']
        elif self.args['dataset_name'] == 'sst2' or self.args['dataset_name'] == 'yelp_polarity':
            verbalizer_predefined = ['negative', 'positive']
        elif self.args['dataset_name'] == 'rte' or self.args['dataset_name'] == 'qnli':
            verbalizer_predefined = ['yes', 'no']
        elif self.args['dataset_name'] == 'mrpc' or self.args['dataset_name'] == 'qqp':
            verbalizer_predefined = ['no', 'yes']
        elif self.args['dataset_name'] == 'boolq':
            verbalizer_predefined = ['no', 'yes']
        elif 'indonlp/NusaX-senti' in self.args['dataset_name']:
            verbalizer_predefined = ['negative', 'neutral', 'positive']
        elif self.args['dataset_name'] == 'ag_news':
            verbalizer_predefined = ['World', 'Sports', 'Business', 'Technology']

        special_space = '▁'
        binary_list = ['SetFit/sst2', 'yelp_polarity', 'SetFit/CR', 'rotten_tomatoes']
        rl_binary_list = ['rl-cr', 'rl-mr', 'rl-sst-2', 
                        'rl-yelp-2']
        if 'bert' in self.args['model_name']:
            special_space = 'Ġ'
            if self.args['dataset_name'] in binary_list:
                verbalizer_predefined = ['terrible', 'great']
            elif self.args['dataset_name'] == 'ag_news':
                verbalizer_predefined = ['World', 'Sports', 'Business', 'Tech']
            elif self.args['dataset_name'] == 'SetFit/sst5' or self.args['dataset_name'] == 'yelp_review_full':
                verbalizer_predefined = ['terrible', 'bad', 'okay', 'good', 'great']
            elif self.args['dataset_name'] in rl_binary_list:
                verbalizer_predefined = ['terrible', 'great']

        verbalizer_predefined = [special_space + v for v in verbalizer_predefined]
        return verbalizer_predefined
    
    def get_data(self, data) -> tuple:
        text_label_list = ['yelp_polarity', 'ag_news', 'SetFit/sst5', 'SetFit/CR', 'rotten_tomatoes', "SetFit/sst2", 'yelp_review_full']
        rl_list = ['rl-agnews', 'rl-cr', 'rl-mr', 'rl-sst-2', 
                        'rl-sst-5', 'rl-yelp-2', 'rl-yelp-5']
        if 'xnli' in self.args['dataset_name'] or self.args['dataset_name'] == 'mnli' or 'anli' in self.args['dataset_name'] or 'americas_nli' in self.args['dataset_name'] or self.args['dataset_name'] == 'snli':
            return [d["premise"] for d in data], [d["hypothesis"] for d in data], [d["label"] for d in data]
        elif self.args['dataset_name'] == 'sst2':
            return [d["sentence"] for d in data], [d["sentence"] for d in data], [d["label"] for d in data]
        elif self.args['dataset_name'] == 'rte' or self.args['dataset_name'] == 'mrpc':
            return [d["sentence1"] for d in data], [d["sentence2"] for d in data], [d["label"] for d in data]
        elif self.args['dataset_name'] == 'qnli':
            return [d["question"] for d in data], [d["sentence"] for d in data], [d["label"] for d in data]
        elif self.args['dataset_name'] == 'qqp':
            return [d["question1"] for d in data], [d["question2"] for d in data], [d["label"] for d in data]
        elif self.args['dataset_name'] == 'boolq':
            return [d["question"] for d in data], [d["passage"] for d in data], [d["label"] for d in data]
        elif 'indonlp/NusaX-senti' in self.args['dataset_name'] or self.args['dataset_name'] in text_label_list:
            return [d["text"] for d in data], [d["text"] for d in data], [d["label"] for d in data]
        elif self.args['dataset_name'] in rl_list:
            return [d["text"] for d in data], [d["text"] for d in data], [d["label"] for d in data]