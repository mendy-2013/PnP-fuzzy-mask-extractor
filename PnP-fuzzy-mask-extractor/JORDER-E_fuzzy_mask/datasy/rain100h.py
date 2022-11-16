import os
from datasy import srdata

class Rain100H(srdata.SRData):
    def __init__(self, args, name='Rain100H', train=True, benchmark=False):
        super(Rain100H, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(Rain100H, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
#         names_mask = [n[self.begin - 1:self.end] for n in names_mask]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(Rain100H, self)._set_filesystem(dir_data)
        self.apath = './dataset/test/100H/'
        print(self.apath)
        self.dir_hr = os.path.join(self.apath, 'norain')
        self.dir_lr = os.path.join(self.apath, 'rain')
#         self.dir_mask = os.path.join(self.apath, 'mask')

