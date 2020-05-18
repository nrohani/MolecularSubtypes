
"""
Implementation of MSDEC using Keras
Author:
Narjes Rohani
We used orginal implementation of DEC by Xifeng Guo. 2017.1.30
"""
#Import packages
#-------------------------------------------------------------------------
import keras
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import keras.backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_samples, silhouette_score
from time import time
from keras import callbacks
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Input
from keras.initializers import VarianceScaling
from keras.engine.topology import Layer, InputSpec
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
import os
from sklearn.cluster import AgglomerativeClustering,SpectralBiclustering,SpectralClustering,KMeans,AgglomerativeClustering,FeatureAgglomeration
from multiprocessing import Pool

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 init='glorot_uniform'):

        super(DEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    # print()
                    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.accuracy_score(self.y, y_pred), metrics.normalized_mutual_info_score(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = x.shape[0] / batch_size * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        import csv
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.accuracy_score(y, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred
def run_pca(propagated_profile):
    
    propagated_profile = propagated_profile.subtract(propagated_profile.mean())
    
    pca = PCA()
    pca.fit(propagated_profile)
    
    propagated_profile_pca = pca.transform(propagated_profile)
    propagated_profile_pca = pd.DataFrame(data=propagated_profile_pca,index=propagated_profile.index)
    PCs=['PC{}'.format(i+1) for i in propagated_profile_pca.columns]
    propagated_profile_pca.columns = PCs
    
    pca_components = pca.components_
    pca_components = pd.DataFrame(data=pca_components,columns=propagated_profile.columns)
    pca_components.index = PCs
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    return propagated_profile_pca, pca_components, explained_variance_ratio


def LoadData():
    train=pd.read_csv("BRCA_training_data_2.csv",delimiter=",")
    trainlabel= pd.read_csv("BRCA_training_lables_2.txt",delimiter="\t")
    test=pd.read_csv("BRCA_validation_data_2.txt",delimiter="\t")
    testlabel=pd.read_csv("BRCA_validation_lables_2.txt",delimiter="\t")
    sim= pd.read_csv("FinalRes.csv",delimiter=",")

    encoder = LabelEncoder()
    train_label = encoder.fit_transform(np.array(trainlabel)[:,1])
    test_label = encoder.fit_transform(np.array(testlabel)[:,1])
    return train,train_label,test,test_label,sim
def run_diffusion(network,rst_prob,mutation_profile,converge_rate,max_iter=100,normalize_mutations=True):
    
    P = renorm(network).T
    if normalize_mutations:
        mutation_profile = renorm(mutation_profile.T).T
    Q = mutation_profile.copy()
    
    for i in range(max_iter):
        Q_new = rst_prob * mutation_profile + (1-rst_prob) * np.dot(Q,P)
        
        delta_Q = Q - Q_new
        delta = np.sqrt(np.sum(np.square(delta_Q)))
        
        print( i,'iteration: delta is',delta)
        
        Q = Q_new
        print(delta)
        if delta < converge_rate:
            break
    
    return Q
def renorm(network):    
    degree = np.sum(network,axis=0)
    return network*1.0/degree  

def run_diffusion_PPR(PPR,mutation_profile,normalize_mutations=False):

    if normalize_mutations:
        mutation_profile = renorm(mutation_profile.T).T

    Q = np.dot(mutation_profile,PPR)

    return Q
# setting the hyper parameters
init = 'glorot_uniform'
pretrain_optimizer = 'adam'
batch_size = 100
maxiter = 2e4
tol = 0.001
save_dir = 'results'

import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

update_interval = 100
pretrain_epochs = 500
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                       distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
#pretrain_optimizer = SGD(lr=1, momentum=0.9)

train,train_label,test,test_label,sim=LoadData()
sim=sim.drop('Name1',axis=1)
train=train.drop('Name1',axis=1)
test=test.drop('Name1',axis=1)

prop=run_diffusion_PPR(sim,train)
proptest=run_diffusion_PPR(sim,test)
prop=np.concatenate((prop, proptest), axis=0)
# prepare the DEC model
y=pd.concat([pd.Series(train_label), pd.Series(test_label)])
data1=np.concatenate((train, test), axis=0)
data=np.concatenate((np.array(data1), np.array(prop)), axis=1)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#X_new = SelectKBest(chi2, k=100).fit_transform(data, y)

def load_network(file_name,output_dir,gene_list=set()):
    import networkx as nx
    # Load graph
    print ("* Loading PPI...")
    with open(file_name) as file_handle:
        if gene_list:
            gene_pairs = [(g1, g2) for g1, g2 in [line.split()[:2] for line in file_handle.read().splitlines()] 
                          if (g1 in gene_list and g2 in gene_list)]
        else:
            gene_pairs = [(g1, g2) for g1, g2 in [line.split()[:2] for line in file_handle.read().splitlines()]]
    
    gene_set = set([g for gene_pair in gene_pairs for g in gene_pair])
    
    G = nx.Graph()
    G.add_nodes_from( gene_set ) # in case any nodes have degree zero
    G.add_edges_from( gene_pairs )
    
    print ("\t- Edges:", len(G.edges()))
    print ("\t- Nodes:", len(G.nodes()))
    
    # Remove self-loops and restrict to largest connected component
    print ("* Removing self-loops, multi-edges, and restricting to")
    print( "largest connected component...")
    selfLoops = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from( selfLoops )
    G = G.subgraph( sorted(nx.connected_components( G ), key=lambda cc: len(cc),
                           reverse=True)[0] )
    nodes = sorted(G.nodes())
    print( "\t- Largest CC Edges:", len( G.edges() ))
    print ("\t- Largest CC Nodes:", len( G.nodes() ))
    
    # Set up output directory
    print ("* Saving updated node list to file...")
    os.system( 'mkdir -p ' + output_dir )
    
    # Index mapping for genes
    index_map = [ "{}\t{}".format(i, nodes[i]) for i in range(len(nodes)) ]
    with open("{}/index_genes".format(output_dir), 'w') as outfile:
        outfile.write( "\n".join(index_map) )
    gene2index = { nodes[i]: i for i in range(len(nodes)) }

    network = nx.to_numpy_matrix( G , nodelist=nodes, dtype=np.float64 )
    network = np.asarray(network)
    return network, gene2index
def create_ppr_matrix(network,rst_prob,output_dir):
    import scipy as sp
    from scipy.linalg import inv
    ## Create the Personalized PageRank (PPR) matrix using Scipy
    # Create "walk" matrix (normalized adjacency matrix)
    print ("* Creating PPR  matrix...")
    W = renorm(network).T
    
    ## Create PPR matrix using Python
    n = network.shape[0]
    PPR = rst_prob*inv(sp.eye(n)-(1.-rst_prob)*W)
    
    os.system( 'mkdir -p ' + output_dir )
    pprfile = "{}/ppr_{:g}.npy".format(output_dir, rst_prob)
    np.save(pprfile, PPR)
    
    return PPR
from scipy.sparse import csr_matrix, csc_matrix, issparse

def load_samples(file_name, nodes, output_dir=''):
    df = pd.read_table(file_name, index_col=0)
    # df should be a sample by node matrix
    samples = df.index
    node_set = set(df.columns)&set(nodes)
    print ("\t- Nodes in adjacency matrix:", len(node_set))
    
    # Index mapping for samples
    sample2index = {samples[i]: i for i in range(len(samples))}
    if output_dir:
        # Set up output directory
        print( "* Saving sample list to file...")
        os.system( 'mkdir -p ' + output_dir)
        index_map = ["{}\t{}".format(i, samples[i]) for i in range(len(samples))]
        with open("{}/index_samples".format(output_dir), 'w') as outfile:
            outfile.write( "\n".join(index_map) )
    
    P_init = pd.DataFrame(index=df.index, columns=nodes)
    P_init.update(df)
    P_init = csr_matrix(P_init.fillna(0).as_matrix())
    return P_init, samples

def load_mutation(file_name,output_dir,gene2index):
    
    with open(file_name) as file_handle:
        pat_gene_pairs = [(p, g) for p, g in [line.split()[:2] for line in file_handle.read().splitlines()] if g in gene2index]
    
    pats = sorted(set([p for p,g in pat_gene_pairs]))
    geneset=set([g for p,g in pat_gene_pairs])
    print ("\t- Genes in adjacency matrix:", len(geneset))
    
    # Set up output directory
    print( "* Saving patient list to file...")
    os.system( 'mkdir -p ' + output_dir )
    
    # Index mapping for genes
    index_map = [ "{}\t{}".format(i, pats[i]) for i in range(len(pats)) ]
    with open("{}/index_patients".format(output_dir), 'w') as outfile:
        outfile.write( "\n".join(index_map) )
    pat2index = { pats[i]: i for i in range(len(pats)) }
    
    mutation_profile = np.zeros((len(pats), len(gene2index)))
    mutation_profile[zip(*[(pat2index[p],gene2index[g]) for p,g in pat_gene_pairs])] = 1.
    return mutation_profile, pat2index
def load_mutation_from_df(df,output_dir,gene2index):
    for gen in df.columns:
        if gen not in gene2index.keys():
            df=df.drop(columns=[str(gen)])
    return df
def load_networkF(file_name, output_dir='', add_selfloop=True):
    # Load a graph
    print ("* Loading network...")
    df = pd.read_table(file_name)
    nfeatures = len(df.columns) - 2
    if add_selfloop:
        df['self_loop'] = 0.
    df['intercept'] = 1.
    node_set = set(df.iloc[:,0]) | set(df.iloc[:,1])
    
    node2index = {}
    nodes = []
    index_map = ''
    selfloop_list = []
    for i, node in enumerate(sorted(list(node_set))):
        node2index[node] = i
        nodes.append(node)
        index_map += '{}\t{}\n'.format(i, node)
        # Add self-loops
        if add_selfloop:
            selfloop_list.append([node, node] + [0.]*nfeatures + [1., 1.])
    if add_selfloop:
        selfloop_df = pd.DataFrame(selfloop_list, columns=df.columns)
        df = pd.concat([df, selfloop_df])

    if output_dir:
        # Set up an output directory
        print ("* Saving node list to file...")
        os.system('mkdir -p ' + output_dir)
        with open("{}/index_nodes".format(output_dir), 'w') as outfile:
            outfile.write( "\n".join(index_map) )
            
    edges = df.iloc[:,:2].applymap(lambda x: node2index[x]).as_matrix()
    features = csc_matrix(df.iloc[:,2:].as_matrix())
    return edges, features, nodes


file_name = 'Data.txt'
edges, features, node_names = load_networkF('BRCA_edge2features_2.txt')
df=pd.read_csv(file_name,delimiter="\t",index_col=0)
file_name = 'PPI.txt'
network_output_dir = 'FI_prop'
network, gene2index = load_network(file_name, network_output_dir,set(df.columns))
rst_prob = 0.3
PPR = create_ppr_matrix(network, rst_prob, network_output_dir)
network_output_dir = 'FI_prop'
PPR = np.load('{}/ppr_0.3.npy'.format(network_output_dir))
output_dir = network_output_dir
mutation_profile=load_mutation_from_df(df,output_dir,gene2index)
usion_PPR(P_init_train,np.array(edges))
pat_diff1 = run_diffusion_PPR(PPR,mutation_profile)
rst_prob = 0.3
converge_rate = 0.0001
M_prop=pd.DataFrame(pat_diff1,columns=mutation_profile.columns, index=df.rename_axis('index',axis=0).index.values)
M_prop.to_csv('propagatedData.csv')
M_prop_pca, pca_components, explained_variance_ratio = run_pca(M_prop)
explained_variance_ratio.tofile('explained_variance_ratio.txt',sep='\n')
print('explained_variance_ratio:',explained_variance_ratio.sum())
prop=pat_diff1

def run_SpectralClustering(args):
    [propagated_profile_pca, n_clusters] = args[:2]
    cluster = SpectralClustering(affinity='nearest_neighbors', n_clusters=n_clusters, n_init=1000, gamma=0.5, 
                                 n_neighbors=10, assign_labels='discretize')
    cluster.fit(propagated_profile_pca)
    print ("Silhouette Score with n_clusters=", n_clusters,"score:", metrics.silhouette_score(propagated_profile_pca, cluster.labels_) )
    return cluster.labels_


def run_KMeans(args):
    [propagated_profile_pca, n_clusters] = args[:2]
    cluster = KMeans(n_clusters=n_clusters, n_init=1000)
    cluster.fit(propagated_profile_pca)
    print ("Silhouette Score with n_clusters=", n_clusters,"score:", metrics.silhouette_score(propagated_profile_pca, cluster.labels_) )
    return cluster.labels_


def run_AgglomerativeClustering(args):
    [propagated_profile_pca, n_clusters] = args[:2]
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='complete')
    cluster.fit(propagated_profile_pca)
    print ("Silhouette Score with n_clusters=", n_clusters,"score:", metrics.silhouette_score(propagated_profile_pca, cluster.labels_) )
    return cluster.labels_


def run_clustering_mp(propagated_profile_pca, maxK, func):
    
    n_processes = maxK-1
    pool = Pool(processes=n_processes)
    
    args = zip([propagated_profile_pca]*(maxK-1),range(2,maxK+1))
    labels = pool.map(func, args)
    
    labels = pd.DataFrame(data=np.array(labels).T,index=propagated_profile_pca.index,
                          columns=['K{}'.format(i) for i in range(2,maxK+1)])
    
    pool.close()
    pool.join()
    
    return labels
propagated_profile_pca=M_prop_pca.iloc[:, :70]


## run network propagation
print(np.array(mutation_profile.shape))
pat_diff2 = run_diffusion(network,rst_prob, np.array(mutation_profile), converge_rate)
print(pat_diff2)
M_prop=pd.DataFrame(pat_diff2,columns=mutation_profile.columns, index=df.rename_axis('index',axis=0).index.values)
M_prop.to_csv('difMPROP2.csv')
#
#idx = np.all(stats.zscore(prop) < 4, axis=1)
#Q1 = pd.DataFrame(prop).quantile(0.02)
#Q3 = pd.DataFrame(prop).quantile(0.98)
#IQR = Q3 - Q1
#idx = ~((pd.DataFrame(prop) < (Q1 - 1.5 * IQR)) | (pd.DataFrame(prop) > (Q3 + 1.5 * IQR))).any(axis=1)
#data = pd.DataFrame(prop).loc[idx]
#y=np.reshape(np.array(pd.DataFrame(y).loc[idx]),(840,))
#data=np.array(data)
#pipe = Pipeline([
#    # the reduce_dim stage is populated by the param_grid
#    ('reduce_dim', 'passthrough'),
#    ('classify', LinearSVC(dual=False, max_iter=1000))
#])
#
#N_FEATURES_OPTIONS = [100,200,90, 80, 70,60,50]
#C_OPTIONS = [1, 10, 100, 1000]
#param_grid = [
#    {
#        'reduce_dim': [PCA(iterated_power=7), NMF()],
#        'reduce_dim__n_components': N_FEATURES_OPTIONS,
#        'classify__C': C_OPTIONS
#    },
#    {
#        'reduce_dim': [SelectKBest(chi2)],
#        'reduce_dim__k': N_FEATURES_OPTIONS,
#        'classify__C': C_OPTIONS
#    },
#]
#reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']
#
#grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid, iid=False)
#grid.fit(data, y)
#
#mean_scores = np.array(grid.cv_results_['mean_test_score'])
## scores are in the order of param_grid iteration, which is alphabetical
#mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
## select score for best C
#mean_scores = mean_scores.max(axis=0)
#bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
#               (len(reducer_labels) + 1) + .5)
#
#plt.figure()
#COLORS = 'bgrcmyk'
#for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
#    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])
#
#plt.title("Comparing feature reduction techniques")
#plt.xlabel('Reduced number of features')
#plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
#plt.ylabel('Digit classification accuracy')
#plt.ylim((0, 1))
#plt.legend(loc='upper left')
#
#plt.show()
#def run_coxph(pat2surv_fn, labels, output_dir):
#    
#    os.system( 'mkdir -p ' + output_dir )
#    
#    pat2surv = pd.read_table(pat2surv_fn,index_col=0)
#    pat2surv = pd.concat([pat2surv,labels],join='inner',axis=1)
#    pat2surv.to_csv('{}/pat2surv2labels.txt'.format(output_dir),sep='\t')
#    
#    os.system("Rscript label2coxph.R {}".format(output_dir))
#    return 0
#run_coxph('pat2clin4surv.txt', labelData, 'survival/')
mutation_profile.to_csv('mutation_profile.csv')
data=prop

dec = DEC(dims=[data.shape[-1], 500, 200], n_clusters=4, init=init)


dec.pretrain(x=data, y=y, optimizer=pretrain_optimizer,
             epochs=pretrain_epochs, batch_size=batch_size,
             save_dir=save_dir)
dec.model.summary()
dec.compile(optimizer=SGD(0.1, 0.3), loss='kld')
y_pred = dec.fit(data, y=y, tol=tol, maxiter=maxiter, batch_size=batch_size,
                 update_interval=update_interval, save_dir=save_dir)
print(silhouette_score(data,y_pred))
labeldc = pd.DataFrame(data=np.array(y_pred).T,index=propagated_profile_pca.index,
                          columns=['K'])
print(labeldc.K.value_counts())
#labeldc.to_csv('NN-proppr-'+str(i)+'.csv')
rf=RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
sfm = SelectFromModel(rf, threshold=0.01)
sfm.fit(M_prop,y_pred)
# Print the name and gini importance of each feature
#for feature in zip(M_prop.columns, rf.feature_importances_):
#    print(feature)
rf.fit(M_prop.head(600),y_pred[:600])
print(accuracy_score(rf.predict(M_prop.tail(263)),labeldc.tail(263)))
for feature_list_index in sfm.get_support(indices=True):
    print(M_prop.columns[feature_list_index])
ttest_fdr_cut = 0.3

#Layer (type)                 Output Shape              Param #   
#=================================================================
#input_1 (InputLayer)         (None, 784)               0         
#_________________________________________________________________
#dense_2 (Dense)              (None, 500)               392500    
#_________________________________________________________________
#dense_3 (Dense)              (None, 500)               250500    
#_________________________________________________________________
#dense_4 (Dense)              (None, 2000)              1002000   
#_________________________________________________________________
#dense_5 (Dense)              (None, 10)                20010     
#_________________________________________________________________
#dense_6 (Dense)              (None, 2000)              22000     
#_________________________________________________________________
#dense_7 (Dense)              (None, 500)               1000500   
#_________________________________________________________________
#dense_8 (Dense)              (None, 500)               250500    
#_________________________________________________________________
#dense_9 (Dense)              (None, 784)               392784    
##=================================================================
#Layer (type)                 Output Shape              Param #   
#=================================================================
#input (InputLayer)           (None, 514)               0         
#_________________________________________________________________
#encoder_0 (Dense)            (None, 500)               257500    
#_________________________________________________________________
#encoder_1 (Dense)            (None, 200)               100200    
#_________________________________________________________________
#clustering (ClusteringLayer) (None, 4)                 800       
