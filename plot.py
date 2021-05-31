import os
import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--arch',type=str)
parser.add_argument('--trial', type=int)
parser.add_argument('--base_dir',type=str,default='/root/volume/AgeDetect/Result')
parser.add_argument('--save_dir',type=str,default='/root/volume/AgeDetect/Plots')
config = parser.parse_args()

chk = torch.load(os.path.join(
                        config.base_dir,config.arch,
                        'latest','model_{}.pt'.format(config.trial)))
if 'epoch' in chk:
    epoch = chk['epoch']
    print("Loaded epoch ", epoch+2)
if 'config' in chk:
    new_config = chk['config']
    print(config)
if 'train_loss' in chk:
    train_loss = chk['train_loss']
    print("Loaded training loss")
if 'val_loss' in chk:
    rmse = chk['val_loss']
    print("Loaded validation loss")
l1_loss = [elem[0] for elem in train_loss]
l2_loss = [elem[1] for elem in train_loss]

plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.plot(range(1,epoch+2),l1_loss,'-r')
plt.xlabel('Epoch')
plt.title('L1 Loss')
plt.subplot(1,3,2)
plt.plot(range(1,epoch+2), l2_loss,'-m')
plt.xlabel('Epoch')
plt.title('L2 Loss')
plt.subplot(1,3,3)
plt.plot(range(1,epoch+2), rmse,'-b')
plt.ylim(5.5, 6.0)
plt.xlabel('Epoch')
plt.title('RMSE Loss')
plt.savefig(os.path.join(config.save_dir,config.arch+'_'+str(config.trial)+'.png'))
print("Saving png at ",str(os.path.join(config.save_dir,config.arch+'_'+str(config.trial)+'.png')))