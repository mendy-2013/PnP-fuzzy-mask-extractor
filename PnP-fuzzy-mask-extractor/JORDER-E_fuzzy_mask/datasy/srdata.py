import os
import glob

from datasy import common
import pickle
import numpy as np
import imageio
import cv2

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        
        data_range = [r.split('-') for r in args.data_range.split('/')]
        # print(data_range)
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        # print(self.begin)
        # print(self.end)
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            # print(path_bin)
            os.makedirs(path_bin, exist_ok=True)

        # print(self.dir_hr)
        list_hr, list_lr = self._scan()
        # print(list_hr)
        # print(list_lr)
        if args.ext.find('bin') >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            list_hr, list_lr ,list_mask= self._scan()
            self.images_hr = self._check_and_load(
                args.ext, list_hr, self._name_hrbin()
            )
            self.images_lr = [
                self._check_and_load(args.ext, l, self._name_lrbin(s)) \
                for s, l in zip(self.scale, list_lr)
            ]
#             self.images_mask = [
#                 self._check_and_load(args.ext, l, self._name_maskbin(s)) \
#                 for s, l in zip(self.scale, list_mask)
#             ]
        else:
            if args.ext.find('img') >= 0 or benchmark:
                self.images_hr, self.images_lr = list_hr, list_lr
            elif args.ext.find('sep') >= 0:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
                for s in self.scale:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{}'.format(s)
                        ),
                        exist_ok=True
                    )
                
                self.images_hr, self.images_lr = [], [[] for _ in self.scale]
                # print(self.images_hr)
                # print(self.images_lr)
                # print(self.images_mask)
                # print(list_hr)
                for h in list_hr:
                    # print(h)
                    b = h.replace(self.apath, path_bin)
                    # print(b)
                    b = b.replace(self.ext[0], '.pt')
                    # print(b)
                    self.images_hr.append(b)
                    self._check_and_load(
                        args.ext, [h], b, verbose=True, load=False
                    )

                for i, ll in enumerate(list_lr):
                    for l in ll:
                        b = l.replace(self.apath, path_bin)
                        b = b.replace(self.ext[1], '.pt')
                        self.images_lr[i].append(b)
                        self._check_and_load(
                            args.ext, [l], b,  verbose=True, load=False
                        )

#                 for j, qq in enumerate(list_mask):
#                     for q in qq:
#                         b = q.replace(self.apath, path_bin)
#                         b = b.replace(self.ext[2], '.pt')
#                         self.images_mask[j].append(b)
#                         self._check_and_load(
#                             args.ext, [q], b,  verbose=True, load=False
#                         )
#             print(self.images_mask)
            print(self.images_hr)
            print(self.images_lr)
        if train:
            # print(len(self.images_hr))
            # print(self.images_hr)
            self.repeat \
                = args.test_every // (len(self.images_hr) // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        # print(self.dir_hr)
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        # print(names_hr)
        names_lr = [[] for _ in self.scale]
#         names_mask = [[] for _ in self.scale]
        # names_lr = sorted(
        #     glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0]))
        # )
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{}{}'.format(
                         filename, self.ext[1]
                    )
                ))
#                 names_mask[si].append(os.path.join(
#                     self.dir_mask, '{}{}'.format(
#                         filename, self.ext[1]
#                     )
#                 ))
        # print(names_lr)

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        # self.apath = './dataset/train/RainTrainL/'
        self.dir_hr = os.path.join(self.apath, 'norain')
        # print(self.dir_hr)
        self.dir_lr = os.path.join(self.apath, 'rain')
#         self.dir_mask = os.path.join(self.apath, 'mask')
        self.ext = ('.png', '.png', '.png')

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR.pt'.format(self.split)
        )

#     def _name_maskbin(self, scale):
#         return os.path.join(
#             self.apath,
#             'bin',
#             '{}_bin_mask.pt'.format(self.split)
#         )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f: ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],
                'image': cv2.imread(_l)
            } for _l in l]
            with open(f, 'wb') as _f: pickle.dump(b, _f)
            return b

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = self.get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr,rgb_range=self.args.rgb_range
        )

        return lr_tensor, hr_tensor, filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
#         f_mask = self.images_mask[self.idx_scale][idx]
#         # print(f_hr)
        # print(f_lr)
        # print(f_mask)

        if self.args.ext.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
#             mask = f_mask['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.args.ext == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
#                 mask = imageio.imread(f_mask)
            elif self.args.ext.find('sep') >= 0:
                with open(f_hr, 'rb') as _f: hr = np.load(_f)[0]['image']
                with open(f_lr, 'rb') as _f: lr = np.load(_f)[0]['image']
#                 with open(f_mask, 'rb') as _f: mask = np.load(_f)[0]['image']

        # print(hr.shape)
        # print(lr.shape)
        # print(mask.shape)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
#                 mask,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih, 0:iw]
            #hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

