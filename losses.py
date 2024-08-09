import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_zoo.loss import lovasz_hinge
from torch.nn.modules.loss import CrossEntropyLoss




__all__ = ['GT_BceDiceLoss_new2']


        
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_zoo.loss import lovasz_hinge
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss

class BCEDiceLoss_newversion(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()


    def forward(self, input, target):


        
        input = torch.sigmoid(input)
        
        
        smooth = 1e-5
        
        num = target.size(0)

      
        input = input.view(num, -1)
        target = target.view(num, -1)
        bce = self.bceloss(input,target)
        intersection = (input * target)
        dice = (2. * intersection.sum(1).pow(2) + smooth) / (input.sum(1).pow(2) + target.sum(1).pow(2) + smooth)
     
        
        dice_loss = 1 - dice.sum() / num
  
        dice1 = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
           
        
        dice_loss1 = 1 - dice1.sum() / num

        return bce +dice_loss+dice_loss1 
         
    
 

   
class GT_BceDiceLoss_new2(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_new2, self).__init__()
        self.bcedice = BCEDiceLoss_newversion()

    def forward(self, pre,out, target, epoch, num_epoch):
        #print(epoch, num_epoch)
        
        bcediceloss = self.bcedice(out, target)
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.6 +self.bcedice(gt_pre0, target) * 0.8
       # print(bcediceloss)
        
        return  (2-torch.sin(torch.tensor(epoch/num_epoch*torch.pi/2)))*(bcediceloss + gt_loss),self.bcedice(gt_pre4, target),self.bcedice(gt_pre3, target),self.bcedice(gt_pre2, target),self.bcedice(gt_pre1, target),self.bcedice(gt_pre0, target)
    
