from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from model.build_gen import Generator, Classifier, DomainClassifier
from datasets.dataset_read import dataset_read
from utils.utils import download
# Training settings
class SolverDANN(object):
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
        download()
        self.data_train, self.data_val, self.data_test = dataset_read(
                source, target, self.batch_size, is_resize = args.is_resize,
                leave_one_num = self.leave_one_num, dataset = args.dataset, 
                sensor_num = args.sensor_num)
        print('load finished!')
        
        self.G = Generator(source=source, target=target, 
                           is_resize = args.is_resize, dataset = args.dataset,
                           sensor_num = args.sensor_num)
        self.LC = Classifier(source=source, target=target,
                             is_resize = args.is_resize, dataset = args.dataset)
        self.DC = DomainClassifier(source=source, target=target,
                             is_resize = args.is_resize, dataset = args.dataset)
        
        if args.eval_only:
            self.data_val = self.data_test            
            self.G = torch.load(r'checkpoint_DANN/best_model_G' + model_name +'.pt')
            self.LC = torch.load(r'checkpoint_DANN/best_model_C1' + model_name + '.pt')
            self.DC = torch.load(r'checkpoint_DANN/best_model_C2' + model_name + '.pt')
            
        self.G.cuda()
        self.LC.cuda()
        self.DC.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)

            self.opt_lc = optim.SGD(self.LC.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_dc = optim.SGD(self.DC.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_lc = optim.Adam(self.LC.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_dc = optim.Adam(self.DC.parameters(),
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_lc.zero_grad()
        self.opt_dc.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.LC.train()
        self.DC.train()
        torch.cuda.manual_seed(1)
        for batch_idx, data in enumerate(self.data_train):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            domain_label_s = torch.zeros(img_s.shape[0])
            domain_label_t = torch.ones(img_t.shape[0])
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = Variable(label_s.long().cuda())
            domain_label_s = Variable(domain_label_s.long().cuda())
            domain_label_t = Variable(domain_label_t.long().cuda())
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()

            feat_s = self.G(img_s)
            output_label_s = self.LC(feat_s)
            loss_label_s = criterion(output_label_s, label_s)
            loss_label_s.backward()
            self.opt_g.step()
            self.opt_lc.step()
            self.reset_grad()

            feat_s = self.G(img_s)
            output_domain_s = self.DC(feat_s)
            feat_t = self.G(img_t)
            output_domain_t = self.DC(feat_t)

            # The objective of the domain classifier is to classify the domain of data accurately.
            loss_domain_s = criterion(output_domain_s, domain_label_s)
            loss_domain_t = criterion(output_domain_t, domain_label_t)
            loss_domain = loss_domain_s + loss_domain_t
            loss_domain.backward()
            self.opt_dc.step()
            self.reset_grad()

            # One objective of the feature generator is to confuse the domain classifier.
            feat_s = self.G(img_s)
            output_domain_s = self.DC(feat_s)
            feat_t = self.G(img_t)
            output_domain_t = self.DC(feat_t)

            loss_domain_s = criterion(output_domain_s, domain_label_s)
            loss_domain_t = criterion(output_domain_t, domain_label_t)
            loss_domain = - loss_domain_s - loss_domain_t
            loss_domain.backward()
            self.opt_g.step()
            self.reset_grad()

            if batch_idx > 500:
                return batch_idx
        return batch_idx

    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.LC.eval()
        self.DC.eval()
        correct = 0.0
        size = 0.0
        for batch_idx, data in enumerate(self.data_val):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            # label = label.squeeze()
            feat = self.G(img)
            output1 = self.LC(feat)
            pred1 = output1.data.max(1)[1]
            k = label.data.size()[0]
            correct += pred1.eq(label.data).cpu().sum()
            size += k
#        if save_model and epoch % self.save_epoch == 0:
#            torch.save(self.G,
#                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
#            torch.save(self.C1,
#                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
#            torch.save(self.C2,
#                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))

        if record_file:
            record = open(record_file, 'a')
            record.write('%s\n' % (float(correct) / size,))
            record.close()
        return float(correct) / size, epoch, size, self.G, self.LC, self.DC
    
    def test_best(self, G, LC, DC):
        G.eval()
        LC.eval()
        DC.eval()
        test_loss = 0
        correct = 0
        size = 0
        for batch_idx, data in enumerate(self.data_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            label = label.squeeze()
            feat = G(img)
            output = LC(feat)
            test_loss += F.nll_loss(output, label).item()
            pred1 = output.data.max(1)[1]
            k = label.data.size()[0]
            correct += pred1.eq(label.data).cpu().sum()
            size += k
        test_loss = test_loss / size
        print('Best test target acc:', 100.0 * correct.numpy() / size,'%')
        return correct.numpy() / size
    
    def calc_correct_ensemble(self, G, LC, DC, x, y):
        x, y = x.cuda(), y.long().cuda()
        x, y = Variable(x, volatile=True), Variable(y)
        y = y.squeeze()
        feat = G(x)
        output = LC(feat)
        pred = output.data.max(1)[1]
        correct_num = pred.eq(y.data).cpu().sum()
        if len(y.data.size()) == 0:
            print('Error, the size of y is 0!')
            return 0, 0
        size_data = y.data.size()[0]
        return correct_num, size_data
    
    def calc_test_acc(self, G, LC, DC, set_name ='T'):
        correct_all = 0
        size_all = 0
        for batch_idx, data in enumerate(self.data_test):
            correct_num, size_data = self.calc_correct_ensemble(
                    G, LC, DC, data[set_name], data[set_name + '_label'])
            if 0 != size_data:
                correct_all += correct_num
                size_all += size_data
        return correct_all.numpy() / size_all
    
    def test_ensemble(self, G, LC, DC):
        G.eval()
        LC.eval()
        DC.eval()
        acc_s = self.calc_test_acc(G, LC, DC, set_name ='S')
        print('Final test source acc:', 100.0 * acc_s,'%')
        acc_t = self.calc_test_acc(G, LC, DC, set_name ='T')
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
