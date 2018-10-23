import argparse
from torchtext import datasets
from torchtext.datasets.babi import BABI20Field
from models.UTransformer import BabiUTransformer
import torch.nn as nn
import torch

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save_path", type=str, default="save/")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--run_avg", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--filter", type=int, default=50)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--emb", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)

    
    return parser.parse_args()

def get_babi_vocab(task):
    text = BABI20Field(70)
    train, val, test = datasets.BABI20.splits(text, root='.data', task=task, joint=False,
                                         tenK=True, only_supporting=False)
    text.build_vocab(train)
    vocab_len = len(text.vocab.freqs) 
    print("VOCAB LEN:",vocab_len )
    return vocab_len + 1

def evaluate(model, criterion, loader):
    model.eval()
    acc = 0
    loss = 0
    for b in loader:
        story, query, answer = b.story,b.query,b.answer.squeeze()
        if(config.cuda): story, query, answer = story.cuda(), query.cuda(), answer.cuda()
        pred_prob = model(story, query)
        loss += criterion(pred_prob[0], answer).item() 
        pred = pred_prob[1].data.max(1)[1] # max func return (max, argmax)
        acc += pred.eq(answer.data).cpu().sum().item() 
    acc  = acc / len(loader.dataset)
    loss = loss / len(loader.dataset)
    return acc,loss

def main(config):
    vocab_len = get_babi_vocab(config.task)
    train_iter, val_iter, test_iter = datasets.BABI20.iters(batch_size=config.batch_size, 
                                                            root='.data', 
                                                            memory_size=70, 
                                                            task=config.task, 
                                                            joint=False,
                                                            tenK=True, 
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
                    filter_size=config.filter)
    print(model)
    if(config.cuda): model.cuda()           
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr=config.lr)

    acc_val, loss_val = evaluate(model, criterion, val_iter)
    print("RAND_VAL ACC:{:.4f}\t RAND_VAL LOSS:{:.4f}".format(acc_val, loss_val))
    correct = 0
    loss_nb = 0
    cnt_batch = 0
    avg_best = 0
    cnt = 0
    model.train()
    for b in train_iter:
        story, query, answer = b.story,b.query,b.answer.squeeze()
        if(config.cuda): story, query, answer = story.cuda(), query.cuda(), answer.cuda()
        opt.zero_grad()
        pred_prob = model(story, query)
        loss = criterion(pred_prob[0], answer)
        loss.backward()
        opt.step()

        ## LOG
        loss_nb += loss.item()
        pred = pred_prob[1].data.max(1)[1] # max func return (max, argmax)
        correct += pred.eq(answer.data).cpu().sum()
        cnt_batch += 1
        if(cnt_batch % 100 == 0):
            acc = correct.item() / float(cnt_batch*config.batch_size)
            loss_nb = loss_nb / float(cnt_batch*config.batch_size)
            print("TRN ACC:{:.4f}\tTRN LOSS:{:.4f}".format(acc, loss_nb))

            acc_val, loss_val = evaluate(model, criterion, val_iter)
            print("VAL ACC:{:.4f}\tVAL LOSS:{:.4f}".format(acc_val, loss_val))

            if(acc_val > avg_best):
                avg_best = acc_val
                cnt = 0
            else:
                cnt += 1
            if(cnt == 10): break
            if(avg_best == 1.0): break 

            correct = 0
            loss_nb = 0
            cnt_batch = 0

    acc_val, loss_val = evaluate(model, criterion, test_iter)
    print("TST ACC:{:.4f}\tTST LOSS:{:.4f}".format(acc_val, loss_val))  

    
if __name__ == "__main__":
    config = parse_config()
    main(config)


