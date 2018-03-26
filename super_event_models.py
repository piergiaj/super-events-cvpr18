import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import temporal_structure_filter as tsf


class SuperEvent(nn.Module):
    def __init__(self, classes=65):
        super(SuperEvent, self).__init__()

        self.classes = classes
        self.dropout = nn.Dropout(0.7)
        self.add_module('d', self.dropout)

        self.super_event = tsf.TSF(3)
        self.add_module('sup', self.super_event)
        self.super_event2 = tsf.TSF(3)
        self.add_module('sup2', self.super_event2)


        # we have 2xD*3
        # we want to learn a per-class weighting
        # to take 2xD*3 to D*3
        self.cls_wts = nn.Parameter(torch.Tensor(classes))
        
        self.sup_mat = nn.Parameter(torch.Tensor(1, classes, 1024))
        stdv = 1./np.sqrt(1024+1024)
        self.sup_mat.data.uniform_(-stdv, stdv)

        self.per_frame = nn.Conv3d(1024, classes, (1,1,1))
        self.per_frame.weight.data.uniform_(-stdv, stdv)
        self.per_frame.bias.data.uniform_(-stdv, stdv)
        self.add_module('pf', self.per_frame)
        
    def forward(self, inp):
        inp[0] = self.dropout(inp[0])
        val = False
        dim = 1
        if inp[0].size()[0] == 1:
            val = True
            dim = 0

        super_event = self.dropout(torch.stack([self.super_event(inp).squeeze(), self.super_event2(inp).squeeze()], dim=dim))
        if val:
            super_event = super_event.unsqueeze(0)
        # we have B x 2 x D*3
        # we want B x C x D*3

        # now we have C x 2 matrix
        cls_wts = torch.stack([torch.sigmoid(self.cls_wts), 1-torch.sigmoid(self.cls_wts)], dim=1)

        # now we do a bmm to get B x C x D*3
        super_event = torch.bmm(cls_wts.expand(inp[0].size()[0], -1, -1), super_event)
        del cls_wts

        # apply the super-event weights
        super_event = torch.sum(self.sup_mat * super_event, dim=2)
        #super_event = self.sup_mat(super_event.view(-1, 1024)).view(-1, self.classes)
        
        super_event = super_event.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        cls = self.per_frame(inp[0])
        return super_event+cls



def get_baseline_model(gpu, classes=65):
    model = nn.Sequential(
        nn.Dropout(0.7),
        nn.Conv3d(1024, classes, (1,1,1)))
    return model.cuda()


def get_tsf_model(gpu, classes=65):
    model = PerFramev4(classes)
    return model.cuda()

