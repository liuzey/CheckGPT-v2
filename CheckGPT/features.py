import torch
import json
import h5py
import time
import argparse
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('domain', type=str)
parser.add_argument('task', type=str)
parser.add_argument('prompt', type=str)
parser.add_argument('--number', type=int, default=0)
parser.add_argument('--gpt', type=int, default=1)
args = parser.parse_args()

domain, task, prompt = args.domain, args.task, args.prompt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large').to(device)
roberta.eval()

data_name = 'gpt_task{}_prompt{}'.format(task, prompt) if args.gpt else 'ground'
feature_name = data_name
if not args.gpt and task == '2':
    feature_name = 'ground_task2'
if args.number:
    feature_name += "_{}".format(args.number)
with open("../GPABench2/{}/{}.json".format(domain, data_name), 'r') as f:
    data = json.load(f)
f.close()

print("Data name: {}, data num: {}".format(data_name, len(list(data.keys()))))
too_long = 0
total_length = args.number if args.number else len(list(data.keys()))

start = time.time()
data = h5py.File('./embeddings/{}.h5'.format(feature_name), 'w')
data.create_dataset('data', (total_length, 512, 1024), dtype='f4')
data.create_dataset('label', (total_length, 1), dtype='i')


def fetch_representation(text):
    tokens = roberta.encode(text)
    last_layer_features = None

    if len(tokens) <= 512:
        last_layer_features = roberta.extract_features(tokens)
    return last_layer_features


i = 0
for abstract in data.values():
    features = fetch_representation(abstract)

    if features is None:
        too_long += 1
        continue

    features_ = F.pad(features, (0, 0, 0, 512 - features.size(1)))

    data["data"][i] = features_.clone().detach().cpu()
    data["label"][i] = torch.ones(1) * (1 - args.gpt)

    if i % 200 == 0:
        print("{}{} at {}th sample. Time used: {}s. Overlong outliers: {}".format(domain, task, i, time.time()-start, too_long))

    i += 1
    if i >= args.number:
        break

data.close()






