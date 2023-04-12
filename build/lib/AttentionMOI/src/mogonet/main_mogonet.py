""" Example for MOGONET classification
"""
from .train_test import train_test


def run_mogonet(num_epoch=200):
    data_folder = 'tmp'
    view_list = [1,2,3]
    view_list = [1]
    num_epoch_pretrain = 500
    num_epoch = num_epoch
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5
    if data_folder == 'tmp':
        num_class = 3
    
    log = train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch)
    
    return log


if __name__ == "__main__":    
    data_folder = 'tmp'
    view_list = [1,2,3]
    view_list = [1]
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5
    if data_folder == 'tmp':
        num_class = 3
    
    train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch)             