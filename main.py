import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from option import  Args, Config
from Dataset import Dataset
from test import test
from Model import Model
from train import train
from utils import Visualizer
import pandas as pd

viz = Visualizer(env='JU-HA')

if __name__ == '__main__':
    args = Args()
    config = Config(args)
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                                batch_size=1, shuffle=False,
                                num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.RAdam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.0005)

    lossList = []
    epoch = 1
    best_auc = -1
    auc = test(test_loader, model, args, viz, device)
    df = pd.DataFrame(columns = ['RoC_AUC', 'PR_AUC', 'Loss', 'Smooth_loss', 'Sparse_loss'], index = range(args.max_epoch))


    iterator = tqdm(range(epoch, args.max_epoch + 1), total = args.max_epoch)

    for step in iterator:
        
        loss = train(train_nloader, train_aloader, model, args.batch_size, optimizer, viz, device)
        auc = test(test_loader, model, args, viz, device)

        iterator.write(f'Epoch {step}/{args.max_epoch}: loss: {loss}, auc: {auc})')
        if step > 0 and best_auc <= auc[0]:
            best_auc = auc[0]
            torch.save(model.state_dict(), './ckpt/' + f'JU-Trans-{step}.pkl')

        df.loc[len(df)] = [auc[0], auc[1], loss[0], loss[1], loss[2]]

    torch.save(model.state_dict(), './ckpt/' + f'JU-Trans-final.pkl')

    df.to_csv('Train_log.csv')