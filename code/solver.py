from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from model.build_gen import Generator, Classifier
from datasets.dataset_read import dataset_read

# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64, source='source',
                 target='target', learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10,
                 leave_one_num = -1, model_name = ''):
        self.args = args
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.leave_one_num = leave_one_num
        
        print('dataset loading')
        self.data_train, self.data_val, self.data_test = dataset_read(
                source, target, self.batch_size, is_resize = args.is_resize,
                leave_one_num = self.leave_one_num, dataset = args.dataset, 
                sensor_num = args.sensor_num)
        print('load finished!')
        
        self.G = Generator(source=source, target=target, 
                           is_resize = args.is_resize, dataset = args.dataset,
                           sensor_num = args.sensor_num)
        self.C1 = Classifier(source=source, target=target, 
                             is_resize = args.is_resize, dataset = args.dataset)
        self.C2 = Classifier(source=source, target=target, 
                             is_resize = args.is_resize, dataset = args.dataset)
        
        if args.eval_only:
            self.data_val = self.data_test            
            self.G = torch.load(r'checkpoint/best_model_G' + model_name +'.pt')
            self.C1 = torch.load(r'checkpoint/best_model_C1' + model_name +'.pt')
            self.C2 = torch.load(r'checkpoint/best_model_C2' + model_name +'.pt')  
            
        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train_souce_only(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.data_train):
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            label_s = Variable(label_s.long().cuda())
            label_s = label_s.squeeze()
            img_s = Variable(img_s)
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            
            # print(label_s.shape)
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s \n' % (loss_s.item()))
                    record.close()
        return batch_idx
    
    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.data_train):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = Variable(label_s.long().cuda())
            label_s = label_s.squeeze()

            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            
            # print(label_s.shape)
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            feat_t = self.G(img_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_dis = self.discrepancy(output_t1, output_t2)
            loss = loss_s - 4 * loss_dis# 1: 92.9; 2: 93.1; 3: 93.5; 5:93.46%
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            for i in range(self.num_k):
                #
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.item(), loss_s1.item(), loss_s2.item()))
                    record.close()
        return batch_idx


    def train_onestep(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.data_train):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = Variable(label_s.long().cuda())
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward(retain_variables=True)
            feat_t = self.G(img_t)
            self.C1.set_lambda(1.0)
            self.C2.set_lambda(1.0)
            output_t1 = self.C1(feat_t, reverse=True)
            output_t2 = self.C2(feat_t, reverse=True)
            loss_dis = -self.discrepancy(output_t1, output_t2)
            #loss_dis.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
               if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.data[0], loss_s1.data[0], loss_s2.data[0]))
                    record.close()
        return batch_idx

    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0.0
        correct1 = 0.0
        correct2 = 0.0
        correct3 = 0.0
        size = 0.0
        for batch_idx, data in enumerate(self.data_val):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            # label = label.squeeze()
            feat = self.G(img)
            output1 = self.C1(feat)
            output2 = self.C2(feat)
            test_loss += F.nll_loss(output1, label).item()
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred2.eq(label.data).cpu().sum()
            correct3 += pred_ensemble.eq(label.data).cpu().sum()
            size += k
        test_loss = test_loss / size
#        if save_model and epoch % self.save_epoch == 0:
#            torch.save(self.G,
#                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
#            torch.save(self.C1,
#                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
#            torch.save(self.C2,
#                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
         
        if record_file:
            record = open(record_file, 'a')
            record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
            record.close()
        return float(correct3) / size, epoch, size, self.G, self.C1, self.C2
    
    def test_best(self, G, C1, C2):
        G.eval()
        C1.eval()
        C2.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0
        for batch_idx, data in enumerate(self.data_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            label = label.squeeze()
            feat = G(img)
            output1 = C1(feat)
            output2 = C2(feat)
            test_loss += F.nll_loss(output1, label).item()
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred2.eq(label.data).cpu().sum()
            correct3 += pred_ensemble.eq(label.data).cpu().sum()
            size += k
        test_loss = test_loss / size
        print('Best test target acc:', 100.0 * correct3.numpy() / size,'%')
        return correct3.numpy() / size
    
    def calc_correct_ensemble(self, G, C1, C2, x, y):
        x, y = x.cuda(), y.long().cuda()
        x, y = Variable(x, volatile=True), Variable(y)
        y = y.squeeze()
        feat = G(x)
        output1 = C1(feat)
        output2 = C2(feat)
        output_ensemble = output1 + output2
        pred_ensemble = output_ensemble.data.max(1)[1]
        correct_num = pred_ensemble.eq(y.data).cpu().sum()
        if len(y.data.size()) == 0:
            return 0, 0
        size_data = y.data.size()[0]
        return correct_num, size_data
    
    def calc_test_acc(self, G, C1, C2, set_name = 'T'):
        correct_all = 0
        size_all = 0
        for batch_idx, data in enumerate(self.data_test):
            correct_num, size_data = self.calc_correct_ensemble(
                    G, C1, C2, data[set_name], data[set_name + '_label'])
            if 0 != size_data:
                correct_all += correct_num
                size_all += size_data
        return correct_all.numpy() / size_all        
    
    def test_ensemble(self, G, C1, C2):
        self.data_train, self.data_val, self.data_test = dataset_read(
            self.source, self.target, 128, self.args.is_resize,
            leave_one_num=self.leave_one_num, dataset=self.args.dataset,
            sensor_num=self.args.sensor_num)
        G.eval()
        C1.eval()
        C2.eval()
        acc_s = self.calc_test_acc(G, C1, C2, set_name = 'S')
        print('Final test source acc:', 100.0 * acc_s,'%')
        acc_t = self.calc_test_acc(G, C1, C2, set_name = 'T')
        print('Final test target acc:', 100.0 * acc_t,'%')
        return acc_s, acc_t
    
    def input_feature(self):
        feature_vec = np.zeros(0)
        label_vec = np.zeros(0)
        domain_vec = np.zeros(0)
        for batch_idx, data in enumerate(self.data_test):
            if data['S'].shape[0] != self.batch_size or \
            data['T'].shape[0] != self.batch_size:
                continue
            if batch_idx > 6:
                break
            
            feature_s = data['S'].reshape((self.batch_size,-1))
            label_s = data['S_label'].squeeze()
            domain_s = np.zeros(label_s.shape)
            
            feature_t = data['T'].reshape((self.batch_size,-1))
            label_t = data['T_label'].squeeze()
            
            domain_t = np.ones(label_t.shape)
            
            feature_c = np.concatenate([feature_s, feature_t])
            
            if 0 == feature_vec.shape[0]:
                feature_vec = np.copy(feature_c)
            else:
                feature_vec = np.r_[feature_vec, feature_c]
            label_c = np.concatenate([label_s, label_t])
            domain_c = np.concatenate([domain_s, domain_t])
            label_vec = np.concatenate([label_vec, label_c])
            domain_vec = np.concatenate([domain_vec, domain_c])
        
        return feature_vec, label_vec, domain_vec
    
    def tsne_feature(self):
        self.G.eval()
        feature_vec = torch.tensor(()).cuda()
        label_vec = np.zeros(0)
        domain_vec = np.zeros(0)
        for batch_idx, data in enumerate(self.data_test):
            if data['S'].shape[0] != self.batch_size or \
            data['T'].shape[0] != self.batch_size:
                continue
            if batch_idx > 6:
                break
            img_s = data['S']
            label_s = data['S_label'].squeeze()
            domain_s = np.zeros(label_s.shape)
            
            img_t = data['T']
            label_t = data['T_label'].squeeze()
            domain_t = np.ones(label_t.shape)
            
            img_c = np.vstack([img_s, img_t])
            
            img_c = torch.from_numpy(img_c)
            img_c = img_c.cuda()    
            img_c = Variable(img_c, volatile=True)
            
            feat_c = self.G(img_c)
            feature_vec = torch.cat((feature_vec, feat_c), 0)
            
            
            label_c = np.concatenate([label_s, label_t])
            domain_c = np.concatenate([domain_s, domain_t])
            label_vec = np.concatenate([label_vec, label_c])
            domain_vec = np.concatenate([domain_vec, domain_c])
            
        return feature_vec.cpu().detach().numpy(), label_vec, domain_vec