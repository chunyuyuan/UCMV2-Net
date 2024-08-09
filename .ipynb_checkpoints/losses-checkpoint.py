import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_zoo.loss import lovasz_hinge
from torch.nn.modules.loss import CrossEntropyLoss




__all__ = ['BCEDiceLossUNEXT','GT_BceDiceLoss','GT_BceDiceLoss_new','GT_BceDiceLoss_a','GT_BceDiceLoss_b','GT_BceDiceLoss_c','GT_BceDiceLoss_ab','GT_BceDiceLoss_ac','GT_BceDiceLoss_bc','GT_BceDiceLoss_abc','GT_BceDiceLoss_new2','GT_BceDiceLoss_new1']


        
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_zoo.loss import lovasz_hinge
from torch.nn.modules.loss import CrossEntropyLoss






##############medt##################
import torch
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss

class BCEDiceLossUNEXT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
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
        
        # Calculate L1 Distance
      #  l1_distance = torch.abs(input - target).sum()/input.numel()
      
        
        dice_loss = 1 - dice.sum() / num
  
        dice1 = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    
        
        
      
        
        dice_loss1 = 1 - dice1.sum() / num

        return bce +dice_loss+dice_loss1 #+l1_distance 
         
    
 
   
class GT_BceDiceLoss(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BCEDiceLoss()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print(len(out[0]))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.3 + self.bcedice(gt_pre1, target) * 0.4 +self.bcedice(gt_pre0, target) * 0.5
        return bcediceloss + gt_loss
    
class GT_BceDiceLoss_new(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_new, self).__init__()
        self.bcedice = BCEDiceLoss_newversion()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print(len(out[0]))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.6 +self.bcedice(gt_pre0, target) * 0.8
        return bcediceloss + gt_loss
class GT_BceDiceLoss_abc(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_abc, self).__init__()
        self.bcedice = BCEDiceLoss_newversion()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print((pre))
        #print(len(pre))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.6 +self.bcedice(gt_pre0, target) * 0.8
       # print(bcediceloss)
        
        return bcediceloss + gt_loss,self.bcedice(gt_pre4, target),self.bcedice(gt_pre3, target),self.bcedice(gt_pre2, target),self.bcedice(gt_pre1, target),self.bcedice(gt_pre0, target)
    
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
    
    
class BCEDiceLoss_a(nn.Module):
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


        return bce #+dice_loss+dice_loss1 #+l1_distance    
    
    
    
class BCEDiceLoss_b(nn.Module):
    def __init__(self):
        super().__init__()
       # self.bceloss = nn.BCELoss()


    def forward(self, input, target):


        
        input = torch.sigmoid(input)
        
        
        smooth = 1e-5
        
        num = target.size(0)

      
        input = input.view(num, -1)
        target = target.view(num, -1)
       # bce = self.bceloss(input,target)
        intersection = (input * target)
        dice = (2. * intersection.sum(1).pow(2) + smooth) / (input.sum(1).pow(2) + target.sum(1).pow(2) + smooth)
        
        # Calculate L1 Distance
      #  l1_distance = torch.abs(input - target).sum()/input.numel()
      
        
        dice_loss = 1 - dice.sum() / num
  
       # dice1 = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
       # dice_loss1 = 1 - dice1.sum() / num

        return dice_loss#+dice_loss1 #+l1_distance   
    
class BCEDiceLoss_c(nn.Module):
    def __init__(self):
        super().__init__()
        #self.bceloss = nn.BCELoss()


    def forward(self, input, target):


        
        input = torch.sigmoid(input)
        
        
        smooth = 1e-5
        
        num = target.size(0)

      
        input = input.view(num, -1)
        target = target.view(num, -1)
       # bce = self.bceloss(input,target)
        intersection = (input * target)
      #  dice = (2. * intersection.sum(1).pow(2) + smooth) / (input.sum(1).pow(2) + target.sum(1).pow(2) + smooth)
        
        # Calculate L1 Distance
      #  l1_distance = torch.abs(input - target).sum()/input.numel()
      
        
       # dice_loss = 1 - dice.sum() / num
  
        dice1 = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice_loss1 = 1 - dice1.sum() / num

        return dice_loss1 #+l1_distance   
    

      
class GT_BceDiceLoss_a(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_a, self).__init__()
        self.bcedice = BCEDiceLoss_a()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print((pre))
        #print(len(pre))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.6 +self.bcedice(gt_pre0, target) * 0.8
       # print(bcediceloss)
        
        return bcediceloss + gt_loss,self.bcedice(gt_pre4, target),self.bcedice(gt_pre3, target),self.bcedice(gt_pre2, target),self.bcedice(gt_pre1, target),self.bcedice(gt_pre0, target)
        
class GT_BceDiceLoss_b(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_b, self).__init__()
        self.bcedice = BCEDiceLoss_b()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print((pre))
        #print(len(pre))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.6 +self.bcedice(gt_pre0, target) * 0.8
       # print(bcediceloss)
        
        return bcediceloss + gt_loss,self.bcedice(gt_pre4, target),self.bcedice(gt_pre3, target),self.bcedice(gt_pre2, target),self.bcedice(gt_pre1, target),self.bcedice(gt_pre0, target)
        
class GT_BceDiceLoss_c(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_c, self).__init__()
        self.bcedice = BCEDiceLoss_c()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print((pre))
        #print(len(pre))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.6 +self.bcedice(gt_pre0, target) * 0.8
       # print(bcediceloss)
        
        return bcediceloss + gt_loss,self.bcedice(gt_pre4, target),self.bcedice(gt_pre3, target),self.bcedice(gt_pre2, target),self.bcedice(gt_pre1, target),self.bcedice(gt_pre0, target)
        
class BCEDiceLoss_ab(nn.Module):
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
        
        # Calculate L1 Distance
      #  l1_distance = torch.abs(input - target).sum()/input.numel()
      
        
        dice_loss = 1 - dice.sum() / num
  


        return bce +dice_loss #+l1_distance 
    
class BCEDiceLoss_ac(nn.Module):
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

  
        dice1 = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    
        
        
      
        
        dice_loss1 = 1 - dice1.sum() / num

        return bce +dice_loss1 #+l1_distance 

    
class BCEDiceLoss_bc(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()


    def forward(self, input, target):


        
        input = torch.sigmoid(input)
        
        
        smooth = 1e-5
        
        num = target.size(0)

      
        input = input.view(num, -1)
        target = target.view(num, -1)
        
        intersection = (input * target)
        dice = (2. * intersection.sum(1).pow(2) + smooth) / (input.sum(1).pow(2) + target.sum(1).pow(2) + smooth)
        
        # Calculate L1 Distance
      #  l1_distance = torch.abs(input - target).sum()/input.numel()
      
        
        dice_loss = 1 - dice.sum() / num
  
        dice1 = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    
        
        
      
        
        dice_loss1 = 1 - dice1.sum() / num

        return dice_loss+dice_loss1 #+l1_distance 

class GT_BceDiceLoss_ab(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_ab, self).__init__()
        self.bcedice = BCEDiceLoss_ab()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print((pre))
        #print(len(pre))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.6 +self.bcedice(gt_pre0, target) * 0.8
       # print(bcediceloss)
        
        return bcediceloss + gt_loss,self.bcedice(gt_pre4, target),self.bcedice(gt_pre3, target),self.bcedice(gt_pre2, target),self.bcedice(gt_pre1, target),self.bcedice(gt_pre0, target)
    
class GT_BceDiceLoss_ac(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_ac, self).__init__()
        self.bcedice = BCEDiceLoss_ac()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print((pre))
        #print(len(pre))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.6 +self.bcedice(gt_pre0, target) * 0.8
       # print(bcediceloss)
        
        return bcediceloss + gt_loss,self.bcedice(gt_pre4, target),self.bcedice(gt_pre3, target),self.bcedice(gt_pre2, target),self.bcedice(gt_pre1, target),self.bcedice(gt_pre0, target)
    
    
class GT_BceDiceLoss_bc(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_bc, self).__init__()
        self.bcedice = BCEDiceLoss_bc()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print((pre))
        #print(len(pre))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.6 +self.bcedice(gt_pre0, target) * 0.8
       # print(bcediceloss)
        
        return bcediceloss + gt_loss,self.bcedice(gt_pre4, target),self.bcedice(gt_pre3, target),self.bcedice(gt_pre2, target),self.bcedice(gt_pre1, target),self.bcedice(gt_pre0, target)
    
    
class GT_BceDiceLoss_new1(nn.Module):
    def __init__(self):
        super(GT_BceDiceLoss_new1, self).__init__()
        self.bcedice = BCEDiceLoss_newversion()

    def forward(self, pre,out, target):
        bcediceloss = self.bcedice(out, target)
        #print(len(out[0]))
        gt_pre4, gt_pre3, gt_pre2, gt_pre1,gt_pre0 = pre
        gt_loss =  self.bcedice(gt_pre4, target) * 0.1 + self.bcedice(gt_pre3, target) * 0.2 + self.bcedice(gt_pre2, target) * 0.3 + self.bcedice(gt_pre1, target) * 0.4 +self.bcedice(gt_pre0, target) * 0.5
        
        return bcediceloss + gt_loss,self.bcedice(gt_pre4, target),self.bcedice(gt_pre3, target),self.bcedice(gt_pre2, target),self.bcedice(gt_pre1, target),self.bcedice(gt_pre0, target)
  