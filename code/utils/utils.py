import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot

def extract_UCI_features(x_mat):
    x_shape = list(x_mat.shape)
    x_shape[-2] = 6
    feature_mat = np.zeros(tuple(x_shape))
    feature_mat[:,:,:,0,:] = np.mean(x_mat, axis = -2)
    feature_mat[:,:,:,1,:] = np.std(x_mat, axis = -2)
    feature_mat[:,:,:,2,:] = np.max(x_mat, axis = -2)
    feature_mat[:,:,:,3,:] = np.min(x_mat, axis = -2)
    feature_mat[:,:,:,4,:] = x_mat[:,:,:,0,:]
    feature_mat[:,:,:,5,:] = x_mat[:,:,:,-1,:]
    return feature_mat

def plot_tsne(data, labels, domains, leave_one_num = 0, is_source_only = False,
              img_path = '../1-results/images/1'):
    tsne = TSNE(perplexity=40, n_components=2, n_iter=300,
                learning_rate=100).fit_transform(data)
    labels = np.squeeze(labels)
    tsne_min, tsne_max = np.min(tsne, 0), np.max(tsne, 0)
    tsne = (tsne - tsne_min) / (tsne_max - tsne_min)
    plt.figure(figsize=(5, 5))
    
    for i in range(tsne.shape[0]):
        plt.text(tsne[i, 0], tsne[i, 1], str(labels[i]), 
                 color=plt.cm.bwr(domains[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})
#        plt.axis('off')
#    if is_source_only:
#        title = 'TSNE before domain adaptation'
#    else:
#        title = 'TSNE after domain adaptation'
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rcParams.update({'font.size': 9})
    plt.show()
    plt.savefig(img_path + '.pdf', bbox_inches='tight')
    return tsne   