import pickle as pkl
import modules.dtset as dtset
import modules.args
import modules.dencoders as endecs
import modules.advecdiffus as warps
import modules.losses as losses
import modules.plots as plot
from modules.meter import AverageMeters
import numpy as np
import pandas as pd

import modules.loading as load



import h5py
import visdom


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# import matplotlib.image as mpimg

from torchvision import transforms

from netCDF4 import Dataset


args = modules.args.parser.parse_args()

viz = visdom.Visdom(env=args.env)

def main():

    hf0 = load.h5py_sandeep()
    # import pdb; pdb.set_trace()
    # print('\n\nth0_torch_tensor', th0.shape)  # for this loading function
    output = open('/Users/mostafa/Desktop/datas/train/data_1.pkl', 'wb')
    pkl.dump(hf0, output)
    output.close()

    # print('>>>> loading the dataset...\n')

    dset = dtset.Dset(args.train_root,
                      seq_len=args.seq_len,
                      target_seq_len=args.target_seq_len,
                      transform=transforms.Compose([transforms.ToTensor()]),
                      )

    test_dset = dtset.Dset(args.test_root,
                           seq_len=args.seq_len,
                           target_seq_len=args.target_seq_len,
                           transform=transforms.Compose([transforms.ToTensor()]),

                           )

    train_indices = range(0, int(len(dset[0]) * args.split))
    val_indices = range(int(len(dset[0]) * args.split), len(dset[0]))

    print('\n\ntrain_indices', train_indices)
    print('\n\nval_indices', val_indices)

    train_loader = DataLoader(dset,
                              batch_size=args.batch_size,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=args.workers,
                              pin_memory=True
                              )

    val_loader = DataLoader(dset,
                            batch_size=args.batch_size,
                            sampler=SubsetRandomSampler(val_indices),
                            num_workers=args.workers,
                            pin_memory=True
                            )

    test_loader = DataLoader(test_dset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True
                             )

    # splits = {
    #     'train': train_loader,
    #     'valid': val_loader,
    #     'test': test_loader
    # }

    splits = {
        'train': train_loader,
    }

    # print('>>>> employing the encode-decode network...\n')
    endecoder = endecs.ConvDeconvEstimator(input_channels=args.seq_len,
                                           upsample_mode=args.upsample)

    # print('>>>> creating warping scheme {}'.format(args.warp))
    warp = warps.__dict__[args.warp]()
    # print('\n>>>> implementing loss function...\n')

    photo_loss = nn.MSELoss()
    smooth_loss = losses.SmoothnessLoss(nn.MSELoss())
    div_loss = losses.DivergenceLoss(nn.MSELoss())
    magn_loss = losses.MagnitudeLoss(nn.MSELoss())

    cudnn.benchmark = True
    optimizer = optim.Adam(endecoder.parameters(), args.lr,
                           betas=(args.momentum, args.beta),
                           weight_decay=args.weight_decay)

    _x, _ys = torch.Tensor(), torch.Tensor()

    viz_wins = {}

    for epoch in range(1, args.epochs + 1):

        results = {}

        for split, dl in splits.items():

            print('split', splits['train'])

            meters = AverageMeters()

            if split == 'train':
                print('\n\ntrain')
                endecoder.train(), warp.train()
            else:
                endecoder.eval(), warp.eval()

            for i, (input, targets) in enumerate(dl):

                # print(f'{i}')
                _x.resize_(input.size())
                _x = input  # .copy_(input)
                # print('_x', _x.size())
                # print('_x', _x[0, 0, 10:14, 10:14])
                _ys.resize_(targets.size())
                _ys = targets  # .copy_(targets)

                # print('input', input.size())
                # print('targets', targets.size())
                print('_x', _x.size())
                print('_ys', _ys.size())

                _ys = _ys.transpose(0, 1).unsqueeze(2)
                print('_ys_transpose_unsqueeze', _ys.size())

                x, ys = Variable(_x), Variable(_ys)
                # x, ys = Variable(_x, requires_grad=True), Variable(_ys, requires_grad=True)

                pl = 0
                sl = 0
                dl = 0
                ml = 0

                ims = []
                ws = []
                # last_im = x[:,:, -1].unsqueeze(1)
                last_im = x[:, -1, :, :].unsqueeze(1)
                # print('last_im', last_im.size())
                #
                # print('x', x.size())
                # print('ys', ys.size())

                for y in _ys:
                    print('\nccc')

                    print('yyy', y.size())
                    print('yyys', _ys.size())

                    # print('x_before_endecoder', x.size())
                    w = endecoder(x)
                    # print('w_size_endecoder(x)', w.size())

                    im = warp(x[:, -1, :, :].unsqueeze(1), w)
                    # print('im_size_warp(last_im_and endecoder(x)',im.size())

                    # print('x_new_before_cat_and_im', x.size())
                    x = torch.cat((x[:, 1:, :, :], im), 1)
                    # print('x_new_after_cat_and_im', x.size())

                    curr_pl = photo_loss(im, y)
                    pl += torch.mean(curr_pl)
                    sl += smooth_loss(w)
                    dl += div_loss(w)
                    ml += magn_loss(w)

                    # ims.append(im.cpu().data.numpy())
                    # ws.append(w.cpu().data.numpy())
                    ims.append(im.data.numpy())
                    ws.append(w.data.numpy())

                pl /= args.target_seq_len
                sl /= args.target_seq_len
                dl /= args.target_seq_len
                ml /= args.target_seq_len

                loss = pl + args.smooth_coef * sl + args.div_coef * dl + args.magn_coef * ml
                print('loss.mean', loss.mean())

                print('loss_data', loss.data)
                print('pl_data', pl.data)
                print('sl_data', sl.data)
                print('dl_data', dl.data)
                print('ml_data', ml.data)

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print('loss_size1', loss.data)

                meters.update(
                    dict(loss=loss.data,
                         pl=pl.data,
                         dl=dl.data,
                         sl=sl.data,
                         ml=ml.data,
                         ),
                    n=x.size(0)
                )

                print('loss_size2', loss.data)

            if not args.no_plot:
                images = [
                    ('target', {'in': input.transpose(0, 1).numpy(), 'out': ys.data.numpy()}),
                    ('im', {'out': ims}),
                    ('ws', {'out': ws}),
                ]

                images = [
                    # ('target', {'in': input.transpose(0, 1).numpy(), 'out': ys.data.numpy()}),
                    ('target', {'in': input.transpose(0, 1).numpy(), 'out': _ys.data.numpy()}),
                    ('ws', {'out': ws}),
                    ('im', {'out': ims}),
                ]

                print('input', input.shape)
                print('ys.data', _ys.data.size())
                print('ims', ims)
                # print('ws', ws.size())

                plt = plot.from_matplotlib(plot.plot_images(images))
                print('plt', plt.shape)
                viz.image(plt.transpose(2, 0, 1),
                          opts=dict(title='{}, epoch {}'.format(split.upper(), epoch)),
                          win=list(splits).index(split),
                          )

            results[split] = meters.avgs()
            print('\n\nEpoch: {} {}: {}\t'.format(epoch, split, meters))

        # transposing the results dict
        res = {}
        legend = []

        for split in results:
            legend.append(split)
            for metric, avg in results[split].items():
                res.setdefault(metric, [])
                res[metric].append(avg)

        # plotting
        for metric in res:
            y = np.expand_dims(np.array(res[metric]), 0)
            x = np.array([[epoch]*len(results)])

            if epoch == 1:
                win = viz.line(X=x, Y=y,
                               opts=dict(showlegend=True,
                                         legend=legend,
                                         title=metric,
                                         )
                               )

                viz_wins[metric] = win

            else:
                viz.line(X=x, Y=y,
                         opts=dict(showlegend=True,
                                   legend=legend,
                                   title=metric,
                                   ),
                         win=viz_wins[metric],
                         update='append',
                         )


if __name__ == '__main__':
    main()




