#this is an example config file for training a HybridPriorWaveletFlow with a HCC prior on the Yuuki simulations

model = "waveletflow"
stepsPerResolution = [8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16]
stepsPerResolution_L = [3] * 8
nLevels = 6     # Number of DWT levels (log_2(nx))
kernel = 3
baseLevel = 1    # Base level for the DWT, baseLevel=1 means that the first DWT level 2x2 pixels
partialLevel = -1 
hiddenChannels = 256
n_res_blocks = 3
actNormScale = 1.0
imShape = [1, 64, 64]  #this is the shape of the training data 
perm = "invconv"
coupling = "rqs"  #or "rqs or checker"
network = ['ConvNet'] * 10
y_classes = None     
y_learn_top = False
y_condition = False
y_weight = None
conditional = True
LU = True
#prior
gauss_priors = [1,2,3,4,5,6] #set this for HCC prior. For WN prior, add all the layers to this list
priorType = 'CC' #Choose 'CC' or 'C'
unnormalize_prior = True

#datset configs 
dataset = 'My_lmdb'   #replace with 'My_lmdb' with your dataset
channels_to_get = ['tsz']
noise_dict = {'tsz': [15.0, 21600]}
data_shape = (5, 64, 64)   #this is the shape of the data in the dataset
dataset_path = '/sdf/group/kipac/users/mati/yuuki_sim_train_64x64/'
val_dataset_path = '/sdf/group/kipac/users/mati/yuuki_sim_val_64x64/'
sample_batch_size = 512
normalize = [True] * 10
norm_type = ['min_max']*10
double_precision = [None, True, True, True, True, False, False]

#powerspectra and normalization
std_path = 'norm_stds/64x64_final_mean_stats_all_levels_tsz.json'
ps_path = 'ps/64x64_tsz_noise_15_noise_freq_21600_tsz_only/dwtlevel'

#output paths
saveDir = '/sdf/group/kipac/users/mati/best_model_64_tsz_rqs/'
plotSaveDir = 'plots_64_tsz_rqs/' 