class Args:
    def __init__(self): 
        self.dataset = 'jmi-ucf'
        self.rgb_list = "./list/jmi-ucf-i3d-train.list"
        self.test_rgb_list = "./list/jmi-ucf-i3d-test.list"
        self.gt = './list/gt-jmi-ucf.npy'
        self.feature_size = 1024
        self.batch_size = 16
        self.max_epoch = 1000
        self.lr = '[0.00001]*1000'

class Config(object):
    def __init__(self, args):
        self.lr = eval(args.lr)
        self.lr_str = args.lr

    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')
