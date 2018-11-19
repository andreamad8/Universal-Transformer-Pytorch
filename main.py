import argparse
from torchtext import datasets
from torchtext.datasets.babi import BABI20Field
from models.UTransformer import BabiUTransformer
from models.common_layer import NoamOpt
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save_path", type=str, default="save/")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--run_avg", type=int, default=10)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--filter", type=int, default=128)
    parser.add_argument("--max_hops", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--emb", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--act", action="store_true")
    parser.add_argument("--act_loss_weight", type=float, default=0.001)
    parser.add_argument("--noam", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_babi_vocab(task):
    text = BABI20Field(70)
    train, val, test = datasets.BABI20.splits(text, root='.data', task=task, joint=False,
                                         tenK=True, only_supporting=False)
    text.build_vocab(train)
    vocab_len = len(text.vocab.freqs) 
    # print("VOCAB LEN:",vocab_len )
    return vocab_len + 1

def evaluate(model, criterion, loader):
    model.eval()
    acc = []
    loss = []
    for b in loader:
        story, query, answer = b.story,b.query,b.answer.squeeze()
        if(config.cuda): story, query, answer = story.cuda(), query.cuda(), answer.cuda()
        pred_prob = model(story, query)
        loss.append(criterion(pred_prob[0], answer).item()) 
        pred = pred_prob[1].data.max(1)[1] # max func return (max, argmax)
        acc.append( pred.eq(answer.data).cpu().numpy() ) 

    acc = np.concatenate(acc)
    acc  = np.mean(acc)
    loss = np.mean(loss)
    return acc,loss

def main(config):
    vocab_len = get_babi_vocab(config.task)
    train_iter, val_iter, test_iter = datasets.BABI20.iters(batch_size=config.batch_size, 
                                                            root='.data', 
                                                            memory_size=70, 
                                                            task=config.task, 
                                                            joint=False,
                                                            tenK=False, 
                                                            only_supporting=False, 
                                                            sort=False, 
                                                            shuffle=True)
    model = BabiUTransformer(num_vocab=vocab_len, 
                    embedding_size=config.emb, 
                    hidden_size=config.emb, 
                    num_layers=config.max_hops,
                    num_heads=config.heads, 
                    total_key_depth=config.depth, 
                    total_value_depth=config.depth,
                    filter_size=config.filter,
                    act=config.act)
    if(config.verbose):
        print(model)
        print("ACT",config.act)
    if(config.cuda): model.cuda()       
    
    criterion = nn.CrossEntropyLoss()
    if(config.noam):
        opt = NoamOpt(config.emb, 1, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        opt = torch.optim.Adam(model.parameters(),lr=config.lr)

    if(config.verbose):
        acc_val, loss_val = evaluate(model, criterion, val_iter)
        print("RAND_VAL ACC:{:.4f}\t RAND_VAL LOSS:{:.4f}".format(acc_val, loss_val))
    correct = []
    loss_nb = []
    cnt_batch = 0
    avg_best = 0
    cnt = 0
    model.train()
    for b in train_iter:
        story, query, answer = b.story,b.query,b.answer.squeeze()
        if(config.cuda): story, query, answer = story.cuda(), query.cuda(), answer.cuda()
        if(config.noam):
            opt.optimizer.zero_grad()
        else:
            opt.zero_grad()
        pred_prob = model(story, query)
        loss = criterion(pred_prob[0], answer)
        if(config.act):
            R_t = pred_prob[2][0] 
            N_t = pred_prob[2][1]
            p_t = R_t + N_t
            avg_p_t = torch.sum(torch.sum(p_t,dim=1)/p_t.size(1))/p_t.size(0)
            loss += config.act_loss_weight * avg_p_t.item()

        loss.backward()
        opt.step()

        ## LOG
        loss_nb.append(loss.item())
        pred = pred_prob[1].data.max(1)[1] # max func return (max, argmax)
        correct.append(np.mean(pred.eq(answer.data).cpu().numpy()))
        cnt_batch += 1
        if(cnt_batch % 10 == 0):
            acc = np.mean(correct)
            loss_nb = np.mean(loss_nb)
            if(config.verbose):
                print("TRN ACC:{:.4f}\tTRN LOSS:{:.4f}".format(acc, loss_nb))

            acc_val, loss_val = evaluate(model, criterion, val_iter)
            if(config.verbose):
                print("VAL ACC:{:.4f}\tVAL LOSS:{:.4f}".format(acc_val, loss_val))

            if(acc_val > avg_best):
                avg_best = acc_val
                weights_best = deepcopy(model.state_dict())
                cnt = 0
            else:
                cnt += 1
            if(cnt == 45): break
            if(avg_best == 1.0): break 

            correct = []
            loss_nb = []
            cnt_batch = 0


    model.load_state_dict({ name: weights_best[name] for name in weights_best })
    acc_test, loss_test = evaluate(model, criterion, test_iter)
    if(config.verbose):
        print("TST ACC:{:.4f}\tTST LOSS:{:.4f}".format(acc_val, loss_val))  
    return acc_test

if __name__ == "__main__":
    config = parse_config()
    for t in range(1,21):
        config.task = t
        acc = []
        for i in range(config.run_avg):
            acc.append(main(config))
        print("Noam",config.noam,"ACT",config.act,"Task:",config.task,"Max:",max(acc),"Mean:",np.mean(acc),"Std:",np.std(acc))

