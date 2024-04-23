import os
import sys
import copy
import h5py
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, ConcatDataset
from configparser import ConfigParser
from logger import Logger

from model import CheckGPT, LSTMwoAttention, RobertaClassificationHead, RobertaMeanPoolingClassificationHead, CNN

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('domain', type=str)
parser.add_argument('task', type=int)
parser.add_argument('prompt', type=int)
parser.add_argument('expid', type=str)
parser.add_argument('--early', type=float, default=100.0, help="Early stopping based on accuracy.")
parser.add_argument('--save', type=int, default=1, help="Set 1 for formal experiments (saving logs, checkpoints, best models). Set 0 for debugging.")
parser.add_argument('--modelid', type=int, default=0, help="0 for CheckGPT, 1 for RCH, 2 for MLP-Pool, 3 for CNN.")
parser.add_argument('--pretrain', type=int, default=0, help="Set 1 for loading pretrained models.")
parser.add_argument('--saved-model', type=str, default=None, help="Path of pretrained models.")
parser.add_argument('--dataamount', type=int, default=100, help="Data size for transfer learning tuning.")
parser.add_argument('--trans', type=int, default=0, help="Set 1 for transfer learning.")
parser.add_argument('--splitr', type=float, default=0.8, help="train-test split ratio.")
parser.add_argument('--ablr', type=float, default=1.0, help="Use 1.0 for full training data.")
parser.add_argument('--lr', type=float, default=0.0002)  # 2e-4 if SGD_OR_ADAM == "adam" else 1e-3
parser.add_argument('--nepochs', type=int, default=100)  # 200 if (TRANSFER or SGD_OR_ADAM == "sgd") else 100
parser.add_argument('--test', type=int, default=0, help="Set 1 for testing.")
parser.add_argument('--mdomain', type=str, default="CS", help="Source model domain.")
parser.add_argument('--mtask', type=int, default=1, help="Source model task.")
parser.add_argument('--mprompt', type=int, default=0, help="Source model prompt.")
parser.add_argument('--mid', type=str, default="00001", help="Source model exp id.")
parser.add_argument('--printall', type=int, default=1, help="Set 0 for brief output.")
parser.add_argument('--v1', type=int, default=0)
parser.add_argument('--adam', type=int, default=1)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batchsize', type=int, default=512)
args = parser.parse_args()

ID = args.expid
domain, task, prompt = args.domain, str(args.task), str(args.prompt)

LOG_INTERVAL = 50
PRINT_ALL = bool(args.printall)
SAVE = bool(args.save)

if not args.test:
    assert not os.path.exists("./exp/{}".format(ID)), "Experiment ID already exists!"
if not os.path.exists("./exp/{}".format(ID)):
    os.mkdir("./exp/{}".format(ID))

SEED = args.seed
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED = bool(args.pretrain)
SAVED_MODEL = args.saved_model
if PRETRAINED:
    assert SAVED_MODEL is not None
    assert os.path.exists(SAVED_MODEL)
TEST = bool(args.test)
FEATURE_DIR = "./embeddings/"

MODELID = args.modelid
LEARNING_RATE = args.lr
N_EPOCH = args.nepochs
EARLYSTOP = args.early
SGD_OR_ADAM = "adam" if args.adam else "sgd"
BATCH_SIZE = args.batchsize
TEST_SIZE = 512 if TEST else BATCH_SIZE

TRANSFER = bool(args.trans)
m_domain, m_task, m_prompt = args.mdomain, str(args.mtask), str(args.mprompt)

if TRANSFER:
    assert m_domain is not None

if SAVE:
    sys.stdout = Logger("./exp/{}/train.log".format(ID), sys.stdout)


class TransNet(nn.Module):
    def __init__(self, fc):
        super(TransNet, self).__init__()
        self.dropout = nn.Dropout(p=0.05)
        self.layers = copy.deepcopy(fc)

    def forward(self, x):
        x = self.layers(self.dropout(x))
        return x

class MyDataset(data.Dataset):
    def __init__(self, archive):
        self.archive = archive
        self.dataset = h5py.File(self.archive, 'r')["data"]
        self.length = len(self.dataset)
        self.labels = h5py.File(self.archive, 'r')["label"]

        index_file = self.archive.replace(".h5", ".index.npy")
        assert index_file != self.archive
        if os.path.exists(index_file):
            self.lengths = np.load(index_file)
        else:
            self.lengths = [np.sum(np.sum(self.dataset[i], axis=1) != 0) for i in range(self.length)]
            self.lengths = [item if item != 0 else 512 for item in self.lengths]
            np.save(index_file, np.array(self.lengths))

        self.lengths = torch.tensor(self.lengths, dtype=torch.int64)

    def __getitem__(self, index):
        x, y = torch.from_numpy(self.dataset[index]).float(), torch.from_numpy(self.labels[index]).long()
        if MODELID == 14:
            x = x.unsqueeze(0)
        return x, y, self.lengths[index]

    def __len__(self):
        return self.length


def save_checkpoint(model, path, optimizer, scheduler, epoch, acc):
    info_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "last_epoch": epoch,
        "best_acc": acc
    }

    torch.save(info_dict, path)
    return


def load_checkpoint(model, path, optimizer, scheduler):
    cp = torch.load(path)
    model.load_state_dict(cp["model_state_dict"], strict=True)
    optimizer.load_state_dict(cp["optimizer_state_dict"])
    scheduler.load_state_dict(cp["lr_scheduler_state_dict"])
    last_epoch = cp["last_epoch"]
    best_acc = cp["best_acc"]
    return last_epoch, best_acc


def load_data(domain, task, size_train=96, size_test=96):
    if True:
        train_dataset_list, test_datasets_list = list(), list()
        domain_list, task_list, prompt_list = [domain], [task], [prompt]
        if args.domain == "ALL":
            domain_list = ["CS", "PHY", "LIT"]
        if args.task == 0:
            task_list = [1, 2, 3]
            ground_list = [os.path.join(FEATURE_DIR, domain_i, "ground.{}").format("h5") for domain_i in domain_list] + \
                          [os.path.join(FEATURE_DIR, domain_i, "ground_task2.{}").format("h5") for domain_i in domain_list]
        elif args.task == 2:
            ground_list = [os.path.join(FEATURE_DIR, domain_i, "ground_task2.{}").format("h5") for domain_i in domain_list]
        else:
            ground_list = [os.path.join(FEATURE_DIR, domain_i, "ground.{}").format("h5") for domain_i in domain_list]
        if args.prompt == 0:
            prompt_list = [1, 2, 3, 4]

        file_list = [os.path.join(FEATURE_DIR, domain_i, "gpt_task{}_prompt{}.{}").format(task_i, prompt_i, "h5")
                     for domain_i in domain_list for task_i in task_list for prompt_i in prompt_list] + ground_list

        for dir_ in file_list:
            data_i = MyDataset(dir_)
            torch.random.manual_seed(SEED)
            train_size = int(args.splitr * len(data_i))
            test_size = len(data_i) - train_size
            train_data_i, test_data_i = torch.utils.data.random_split(data_i, [train_size, test_size])
            abl_train_size = int(args.ablr * train_size)
            if TRANSFER:
                abl_train_size = args.dataamount
            train_data_i, _ = torch.utils.data.random_split(train_data_i, [abl_train_size, train_size - abl_train_size])
            if not TEST:
                if "prompt" in dir_:
                    # if args.domain == "ALL":
                    #     # Sample 1/3 of the data from each domain
                    #     train_data_i = torch.utils.data.Subset(train_data_i, list(range(0, len(train_data_i), 3)))
                    #     test_data_i = torch.utils.data.Subset(test_data_i, list(range(0, len(test_data_i), 3)))
                    # if args.task == 0:
                    #     # Sample 1/3 of the data from each task
                    #     train_data_i = torch.utils.data.Subset(train_data_i, list(range(0, len(train_data_i), 3)))
                    #     test_data_i = torch.utils.data.Subset(test_data_i, list(range(0, len(test_data_i), 3)))
                    if args.prompt == 0:
                        # Sample 1/2 of the data from each prompt
                        train_data_i = torch.utils.data.Subset(train_data_i, list(range(0, len(train_data_i), 2)))
                        test_data_i = torch.utils.data.Subset(test_data_i, list(range(0, len(test_data_i), 2)))

                if "ground" in dir_:
                    # if args.domain == "ALL":
                    #     # Sample 1/3 of the data from each domain
                    #     train_data_i = torch.utils.data.Subset(train_data_i, list(range(0, len(train_data_i), 3)))
                    #     test_data_i = torch.utils.data.Subset(test_data_i, list(range(0, len(test_data_i), 3)))
                    # if args.task == 0:
                    #     train_data_i = torch.utils.data.Subset(train_data_i, list(range(0, len(train_data_i), 2)))
                    #     test_data_i = torch.utils.data.Subset(test_data_i, list(range(0, len(test_data_i), 2)))
                    pass

            train_dataset_list.append(train_data_i)
            test_datasets_list.append(test_data_i)
        train_data = ConcatDataset(train_dataset_list)
        test_data = ConcatDataset(test_datasets_list)

    train_loader = DataLoader(dataset=train_data,
                              num_workers=4,
                              batch_size=size_train,
                              drop_last=True,
                              shuffle=True,
                              pin_memory=True,
                              prefetch_factor=2)
    test_loader = DataLoader(dataset=test_data,
                             num_workers=4,
                             batch_size=size_test,
                             drop_last=False,
                             shuffle=False,
                             pin_memory=True,
                             prefetch_factor=2)
    return train_loader, test_loader


def test(model, dataloader, epoch, print_freq=1):
    model.eval()
    n_correct, correct_0, correct_1, sum_0, sum_1 = 0, 0, 0, 0, 0
    start = time.time()

    with torch.no_grad():
        for i, (t_img, t_label, t_length) in enumerate(dataloader):
            t_img, t_label, t_length = t_img.to(DEVICE), t_label.to(DEVICE).squeeze(1), t_length
            class_output = model(t_img, t_length)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == t_label).sum().item()
            correct_0 += ((pred[1] == t_label) * (t_label == 0)).sum().item()
            correct_1 += ((pred[1] == t_label) * (t_label == 1)).sum().item()
            sum_0 += (t_label == 0).sum().item()
            sum_1 += (t_label == 1).sum().item()
            if i % (LOG_INTERVAL*print_freq) == 0 and PRINT_ALL and not TRANSFER:
                print('Batch: [{}/{}], Time used: {:.4f}s'.format(i, len(dataloader), time.time() - start))

    accu = float(n_correct) / len(dataloader.dataset) * 100
    accu_0 = float(correct_0) / sum_0 * 100
    accu_1 = float(correct_1) / sum_1 * 100
    tp, fp, tn, fn = correct_0, sum_1 - correct_1, correct_1, sum_0 - correct_0
    if tp + fp == 0:
        precision = 0
    else:
        precision = float(tp) / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = float(tp) / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    if PRINT_ALL:
        print('{}{}， Epoch:{}, Test accuracy: {:.4f}%, Acc_GPT: {:.4f}%, Acc_Human: {:.4f}%, F1: {:.4f}'.format(
            domain, task, epoch, accu, accu_0, accu_1, f1))
    return accu, accu_0, accu_1, f1


def train(model, optimizer, scheduler, dataloader, test_loader):
    loss_class = torch.nn.CrossEntropyLoss()
    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    if TRANSFER:
        print("Transfer Learning. s: {}, t: {}".format(m_domain+m_task+m_prompt, domain+task+prompt))
        best_acc, _, _, _ = test(model, test_loader, 0)
        print("Acc without fine-tuning: {:.4f}".format(best_acc))

    last_epoch, best_acc = 0, 0

    len_dataloader = len(dataloader)
    acc = 0
    not_increase = 0
    for epoch in range(last_epoch + 1, N_EPOCH + 1):
        start = time.time()
        model.train()
        data_iter = iter(dataloader)
        n_correct = 0

        i = 1
        while i < len_dataloader + 1:
            data_source = next(data_iter)
            optimizer.zero_grad()

            img, label, length = data_source[0].to(DEVICE), data_source[1].to(DEVICE).squeeze(1), data_source[2]

            class_output = model(img, length)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == label).sum().item()
            err = loss_class(class_output, label)
            err.backward()
            optimizer.step()
            scheduler.step()

            # scaler.scale(err).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if i % LOG_INTERVAL == 0 and PRINT_ALL:
                print(
                    'Epoch: [{}/{}], Batch: [{}/{}], Err: {:.4f}, Time used: {:.4f}s'.format(
                        epoch, N_EPOCH, i, len_dataloader, err.item(), time.time() - start,
                        ))
            i += 1
            # torch.cuda.empty_cache()

        accu = float(n_correct) / (len(dataloader.dataset)) * 100
        if PRINT_ALL:
            print('{}_TASK{}_PROMPT{}， Epoch:{}, Train accuracy: {:.4f}%'.format(domain, task, prompt, epoch, accu))

        if not TRANSFER and SAVE:
            save_checkpoint(model, "./exp/{}/Checkpoint_{}_Task{}.pth".format(ID, domain, task),
                            optimizer, scheduler, epoch, best_acc)
        old_acc = acc
        scheduler.step()

        if epoch % 1 == 0:
            acc, _, _, _ = test(model, test_loader, epoch)
            if acc <= best_acc:
                not_increase += 1
            else:
                not_increase = 0

            if acc > best_acc:
                best_acc = acc
                if TRANSFER:
                    name = "./exp/{}/Best_s_{}{}{}_t_{}{}{}.pth".format(ID, m_domain, m_task, m_prompt, domain, task, prompt)
                else:
                    name = "./exp/{}/Best_{}_Task{}.pth".format(ID, domain, task)
                if SAVE:
                    torch.save(model.state_dict(), name)
                if not TRANSFER:
                    print("Best model saved.")

            limit = 8
            if best_acc >= 100.00 or (epoch >= 5 and ((EARLYSTOP > 0 and best_acc >= EARLYSTOP) or (not_increase >= limit))):
                break

    return best_acc


if __name__ == '__main__':
    print(args)
    torch.random.manual_seed(SEED)

    train_loader, test_loader = load_data(domain, task, size_train=BATCH_SIZE, size_test=TEST_SIZE)
    rnn = None
    if MODELID == 0:
        rnn = CheckGPT(input_size=1024, hidden_size=128, batch_first=True, dropout=0.5, bidirectional=True,
                              num_layers=2, device=DEVICE_NAME, v1=args.v1).to(DEVICE)
    elif MODELID == 1:
        rnn = LSTMwoAttention(input_size=1024, hidden_size=128, batch_first=True, dropout=0.5, bidirectional=True,
                              num_layers=2, device=DEVICE_NAME, v1=args.v1).to(DEVICE)
    # for ablation study
    elif MODELID == 2:
        rnn = RobertaClassificationHead().to(DEVICE)
    elif MODELID == 3:
        rnn = RobertaMeanPoolingClassificationHead().to(DEVICE)
    elif MODELID == 4:
        rnn = CNN().to(DEVICE)

    assert rnn is not None

    if PRETRAINED or TEST:
        rnn.load_state_dict(torch.load(SAVED_MODEL), strict=True)
        try:
            rnn.load_state_dict(torch.load(SAVED_MODEL), strict=True)
        except RuntimeError:
            rnn = CheckGPT(input_size=1024, hidden_size=256, batch_first=True, dropout=0.5, bidirectional=True,
                           num_layers=2, device=DEVICE_NAME, v1=args.v1).to(DEVICE)
            rnn.load_state_dict(torch.load(SAVED_MODEL), strict=True)
        # torch.save(rnn.state_dict(), "./exp/{}/Best_{}_Task{}.pth".format(ID, domain, task))

    model = rnn

    if not TEST:
        for param in model.parameters():
            param.requires_grad = True

    if TRANSFER:
        rnn.load_state_dict(torch.load("./exp/{}/Best_{}_Task{}.pth".format(args.mid, m_domain, m_task)), strict=False)
        model.fc = TransNet(model.fc)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        if SGD_OR_ADAM == "adam":
            optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE)
        else:
            optimizer = optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    else:
        if SGD_OR_ADAM == "adam":
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        else:
            optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCH)

    if not TEST:
        best_acc = train(model, optimizer, scheduler, train_loader, test_loader)

    best_acc, _, _, _ = test(model, test_loader, 0)

    if TRANSFER:
        print("Transfer Learning, S: {}, T: {}, Acc: {:.4f}%".format(m_domain+m_task+m_prompt, domain+task+prompt, best_acc))
