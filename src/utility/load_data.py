import torch.utils.data as data
import json
import os
import pickle
from src.utility.util import *
from src.utility.load_data import *
from src.utility.ml_gcn import *
from src.utility.engine_test import *

class load_data(data.Dataset):
    def __init__(self, root, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)

        self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno_custom_final_0.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category_custom_final.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])

        img = Image.open(filename).convert('RGB')
        
        model = gcn_resnet101(num_classes=10, t=0.03, adj_file='../data/kfashion_style/custom_adj_final.pkl')
    
        normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
        self.transform = transforms.Compose([
                Warp(224),
                transforms.ToTensor(),
                normalize,
            ])

        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.inp), target
