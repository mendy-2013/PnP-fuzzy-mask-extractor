import torch

import utility
import data
import modelsy
import loss
from option import args
from trainer_reslstm import Trainer
import os
from reslstm import reslstm

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = modelsy.Model(args, checkpoint)
    model1 = reslstm().cuda()
    ckp_path = os.path.join(args.save_path_mask, 'latest')
    obj = torch.load(ckp_path)
    model1.load_state_dict(obj['net'])

    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(pytorch_total_params)

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, model1, loss, checkpoint)

    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()
