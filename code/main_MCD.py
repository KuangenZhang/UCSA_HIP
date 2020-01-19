from __future__ import print_function
import argparse
import torch
from solver import Solver
from utils.utils import *
import os
import numpy as np
import time


def classify(args, leave_one_num = -1):
    
    if args.is_source_only:
        model_name = '_' + args.dataset + '_sensor_num' + str(args.sensor_num) + '_subject_idx' + str(leave_one_num) + '_souce_only'
    else:
        model_name = '_' + args.dataset + '_sensor_num' + str(args.sensor_num) + '_subject_idx' + str(leave_one_num)
    
    # if not args.one_step:
    torch.cuda.empty_cache()

    solver = Solver(args, source=args.source, target=args.target, 
                    learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, num_k=args.num_k, 
                    all_use=args.all_use,
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch, 
                    leave_one_num = leave_one_num, model_name = model_name)
    record_num = 0
    best_acc = 0
    record_train = 'record/%s_%s_k_%s_onestep_%s_%s.txt' % (
        args.source, args.target, args.num_k, args.one_step, record_num)
    record_test = 'record/%s_%s_k_%s_onestep_%s_%s_test.txt' % (
        args.source, args.target, args.num_k, args.one_step, record_num)
    
    
    while os.path.exists(record_train):
        record_num += 1
        record_train = 'record/%s_%s_k_%s_onestep_%s_%s.txt' % (
            args.source, args.target, args.num_k, args.one_step, record_num)
        record_test = 'record/%s_%s_k_%s_onestep_%s_%s_test.txt' % (
            args.source, args.target, args.num_k, args.one_step, record_num)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists('record'):
        os.mkdir('record')
    if not args.eval_only:
        count = 0
        for t in range(args.max_epoch):
            if args.is_source_only:
                num = solver.train_souce_only(t, record_file=record_train)
            else:
                if not args.one_step:
                    num = solver.train(t, record_file=record_train)
                else:
                    num = solver.train_onestep(t, record_file=record_train)
            count += num
            if t % 1 == 0:
                acc, epoch, size, G, C1, C2 = solver.test(
                        t, record_file=record_test, save_model=args.save_model)
                is_best = acc > best_acc
                best_acc = max(acc, best_acc)
                if is_best:
                    print('Best validation acc:', 100.0 * best_acc,'%')
                    torch.save(G, '%s/best_model_G' % (args.checkpoint_dir) + model_name +'.pt')
                    torch.save(C1,'%s/best_model_C1' % (args.checkpoint_dir) + model_name +'.pt')
                    torch.save(C2,'%s/best_model_C2' % (args.checkpoint_dir) + model_name +'.pt')   
                if count >= 20000:
                    break
    else:
        print('Only evaluate.')
        # start = time.time()
        # acc, epoch, size, G, C1, C2 = solver.test(0)
        # end = time.time()
        # print('Test acc:{}%, forward time: {} s.'.format(100.0 * acc, (end-start)/size))
        #        # uncomment the following tsne code to plot tsne figures
        #        global features, labels, domains, tsne
        #        if args.is_source_only:
        #            features, labels, domains = solver.input_feature()
        #        else:
        #            features, labels, domains = solver.tsne_feature()
        #
        #        tsne = plot_tsne(features, labels.astype(np.int), domains, leave_one_num = 0,
        #                  is_source_only = args.is_source_only,
        #                  img_path = 'results/images/' + model_name)
            
    G = torch.load(r'checkpoint/best_model_G' + model_name +'.pt')
    C1 = torch.load(r'checkpoint/best_model_C1' + model_name +'.pt')
    C2 = torch.load(r'checkpoint/best_model_C2' + model_name +'.pt')
    return solver.test_ensemble(G, C1, C2)
    

def main(args):
    if 'UCI' == args.dataset:
        sub_num = 8
    elif 'NW' == args.dataset:
        sub_num = 10
    acc_s_MCD = np.zeros(sub_num)
    acc_t_MCD = np.zeros(sub_num)
    for i in range(sub_num):
        print('Test ', i)
        acc_s_MCD[i], acc_t_MCD[i] = classify(args, leave_one_num = i)
    print ('MCD: mean of test acc in the source domain:', np.mean(acc_s_MCD))
    print ('MCD: mean of test acc in the target domain:', np.mean(acc_t_MCD))
    return np.transpose(np.c_[acc_s_MCD, acc_t_MCD])

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NW', metavar='N',
                        help='Dataset is NW or UCI?')
    parser.add_argument('--sensor_num', type=int, default=0, metavar='N',
                        help='Different combination of sensors: \
                        0: all sensors; \
                        1: NW-EMG or UCI-torso; \
                        2: NW-IMU or UCI-right arm; \
                        3: NW-Angle or UCI-left arm; \
                        4: NW-EMG + IMU or UCI-right leg; \
                        5: NW-EMG + Angle or UCI-left leg; \
                        6: NW-IMU + Angle')
    parser.add_argument('--is_source_only', type=bool, default=False,
                        help='Source only means no domain adaptation?')
    parser.add_argument('--all_use', type=str, default='no', metavar='N',
                        help='use all training data? in usps adaptation')
    parser.add_argument('--is_resize', type=bool, default=True,
                        help='Resize the data to 2D image?')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--checkpoint_dir', type=str, default=r'checkpoint', metavar='N',
                        help='source only or not')
    parser.add_argument('--eval_only', default=False,
                        help='evaluation only option')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--max_epoch', type=int, default=50, metavar='N',
                        help='how many epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num_k', type=int, default=4, metavar='N',
                        help='hyper paremeter for generator update')
    parser.add_argument('--one_step', action='store_true', default=False,
                        help='one step training with gradient reversal layer')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
    parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                        help='epoch to resume')
    parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                        help='when to restore the model')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='save_model or not')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--source', type=str, default='source', metavar='N',
                        help='source dataset')
    parser.add_argument('--target', type=str, default='target', metavar='N', help='target dataset')
    parser.add_argument('--use_abs_diff', action='store_true', default=False,
                        help='use absolute difference value as a measurement')

    args = parser.parse_args()
    args.eval_only = (args.eval_only == 'True')

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print(args)

    # args.dataset = 'UCI'
    # args.is_source_only = False
    #
    # acc_MCD_uci = main(args)
    # np.savetxt("results/acc_MCD_all_sensor_uci.csv",
    #               acc_MCD_uci, delimiter=",")

    # args.is_source_only = True
    #
    # acc_MCD_uci = main(args)
    # np.savetxt("results/acc_MCD_all_sensor_uci_souce_only.csv",
    #               acc_MCD_uci, delimiter=",")

    args.dataset = 'NW'
    args.is_source_only = False
    
    acc_MCD_nw = main(args)
    np.savetxt("results/acc_MCD_all_sensor_nw.csv",
              acc_MCD_nw, delimiter=",")

    # args.is_source_only = True
    #
    # acc_MCD_nw = main(args)
    # np.savetxt("results/acc_MCD_all_sensor_nw_souce_only.csv",
    #           acc_MCD_nw, delimiter=",")

    # args.is_source_only = False
    # args.dataset = 'NW'
    #
    # acc_MCD_nw = np.zeros((12,10))
    # for i in range(6):
    #     args.sensor_num = i + 1
    #     acc_MCD_nw[2*i:2*(i+1), :] = main(args)
    # np.savetxt("results/acc_MCD_compare_sensor_nw.csv",
    #           acc_MCD_nw, delimiter=",")
    #
    # args.dataset = 'UCI'
    #
    # acc_MCD_uci = np.zeros((10,8))
    # for i in range(5):
    #     args.sensor_num = i + 1
    #     acc_MCD_uci[2*i:2*(i+1), :] = main(args)
    # np.savetxt("results/acc_MCD_compare_sensor_uci.csv",
    #           acc_MCD_uci, delimiter=",")
    