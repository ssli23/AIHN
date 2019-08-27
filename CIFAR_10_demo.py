import CIFAR_10 as dpsh
import pickle
from datetime import datetime
import torch

def CIFAR_10_demo():
    lamda = 10
    param = {}
    param['lambda'] = lamda

    gpu_ind = {0,1}
    print(torch.cuda.is_available())
    bits = [12, 24, 32, 48]
    # bits = [1000]
    for bit in bits:
        filename = 'log/AIHN' + str(bit) + 'bits_CIFAR-10' + '_' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.pkl'
        param['filename'] = filename
        print('---------------------------------------')
        print('[#bit: %3d]' % (bit))
        result = dpsh.algo(bit, param, gpu_ind)
        print('[MAP: %3.5f]' % (result['map']))
        print('---------------------------------------')
        fp = open(result['filename'], 'wb')
        pickle.dump(result, fp)
        fp.close()

if __name__=="__main__":
    CIFAR_10_demo()
