import sys

sys.path.append('../loader')
from datasets.unaligned_data_loader import UnalignedDataLoader
from utils import FileIO

def dataset_read(source, target, batch_size, is_resize = False, 
                 leave_one_num = -1, dataset = 'NW', 
                sensor_num = 0):
    S_train = {}
    S_val = {}
    S_test = {}
    T_train = {}
    T_val = {}
    T_test = {}
    
    if 'NW' == dataset:
        x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
        x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
        FileIO.load_st_AB_mat(data_path = 'data/AB_dataset/AB_', X_dim = 4, 
                              is_resize = is_resize, 
                              leave_one_num = leave_one_num, 
                              sensor_num = sensor_num)
    elif 'UCI' == dataset:
        x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
        x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
        FileIO.load_UCI_mat(data_path = 'data/1_dataset_UCI_DSADS/Features/',
                        feature_length = 6*45, X_dim = 4, 
                        is_resize = is_resize, leave_one_num = leave_one_num,
                        sensor_num = sensor_num)
    
    S_train['imgs'] = x_s_train
    S_train['labels'] = y_s_train
    T_train['imgs'] = x_t_train
    T_train['labels'] = y_t_train

    # input target samples for both 
    S_val['imgs'] = x_s_val
    S_val['labels'] = y_s_val
    T_val['imgs'] = x_t_val
    T_val['labels'] = y_t_val

    S_test['imgs'] = x_s_test
    S_test['labels'] = y_s_test

    T_test['imgs'] = x_t_test
    T_test['labels'] = y_t_test
        
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S_train, T_train, batch_size, batch_size)
    # train_loader.initialize(T_train, S_train, batch_size, batch_size)
    data_train = train_loader.load_data()
    
    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_val, T_val, batch_size, batch_size)
    # test_loader.initialize(T_val, S_val, batch_size, batch_size)
    data_val = test_loader.load_data()
    
    final_test_loader = UnalignedDataLoader()
    final_test_loader.initialize(S_test, T_test, batch_size, batch_size)
    # final_test_loader.initialize(T_test, S_test, batch_size, batch_size)
    data_test = final_test_loader.load_data()
    
    return data_train, data_val, data_test
