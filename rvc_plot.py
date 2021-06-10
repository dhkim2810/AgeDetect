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

plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.plot(range(1,epoch+2),train_loss[0],'-r')
plt.xlabel('Epoch')
plt.title('L1 Loss')
plt.subplot(1,3,2)
plt.plot(range(1,epoch+2),train_loss[1],'-r')
plt.xlabel('Epoch')
plt.title('KL Divergence')
plt.subplot(1,3,3)
plt.plot(range(1,epoch+2), rmse,'-b')
plt.xlabel('Epoch')
plt.title('RMSE')
plt.savefig(os.path.join(config.save_dir,config.arch+'_'+str(config.trial)+'.png'))
print("Saving png at ",str(os.path.join(config.save_dir,config.arch+'_'+str(config.trial)+'.png')))