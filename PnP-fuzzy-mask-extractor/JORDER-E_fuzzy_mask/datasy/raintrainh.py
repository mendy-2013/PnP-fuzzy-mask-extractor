import os
from datasy import srdata

class RainTrainH(srdata.SRData):
    def __init__(self, args, name='RainTrainH', train=True, benchmark=False):
        super(RainTrainH, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(RainTrainH, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
#         names_mask = [n[self.begin - 1:self.end] for n in names_mask]
        # print(names_hr)
        # print(names_lr)
        # print(names_mask)
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(RainTrainH, self)._set_filesystem(dir_data)
        self.apath = './dataset/train/100H/'

        # print(self.apath)
        self.dir_hr = os.path.join(self.apath, 'norain')
        self.dir_lr = os.path.join(self.apath, 'rain')
#         self.dir_mask = os.path.join(self.apath, 'mask')
        # print(self.dir_mask)

