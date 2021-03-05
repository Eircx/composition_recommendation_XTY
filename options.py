import argparse
from os import path, makedirs
import torch

def printInfo(inputs):
    print("[ XTY's StyleGAN ] %s" % (inputs))

def printParameters(args_dict):
    printInfo("+++++++++++  Parameters ++++++++++++")
    for key in sorted(args_dict.keys()):
        printInfo("{:<20} : {}".format(key, args_dict[key]))
    printInfo("++++++++++++++++++++++++++++++++++++")

class TrainOptions():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch StyleGAN Traning")
        parser.add_argument("--outf", type=str, default="train_result", help="folder to output models")
        parser.add_argument("--save-checkpoints", type=int, default=10, help="print frequency (default: 50)")
        parser.add_argument("--exp-id", type=str, default="1", help="record the id of experiment")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--epoches", type=int, default=200)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--cuda", default=0, type=int, help="choose which GPU to use in training")
        parser.add_argument("--multi-gpu", type=int, default=4, help="if --multi-gpu<=1 then use only one GPU!")
        parser.add_argument("--resume", type=int, default=0, help="whether to resume training")
        self.opts = parser.parse_args()
        

    def parse(self):
        self.opts.device = 'cuda:{}'.format(self.opts.cuda) if torch.cuda.is_available() else "cpu"
        self.opts.model_dir = path.join(path.dirname(__file__), self.opts.outf, self.opts.exp_id, "models")
        self.opts.log_dir = path.join(path.dirname(__file__), self.opts.outf, self.opts.exp_id, "logs")
        self.opts.img_dir = path.join(path.dirname(__file__), self.opts.outf, self.opts.exp_id, "images")
        if not path.exists(self.opts.model_dir):
            makedirs(self.opts.model_dir)
        if not path.exists(self.opts.log_dir):
            makedirs(self.opts.log_dir)
        if not path.exists(self.opts.img_dir):
            makedirs(self.opts.img_dir)
        printParameters((self.opts.__dict__))
        return self.opts
        
        
