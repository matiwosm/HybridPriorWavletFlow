#this is an example config file for training a HybridPriorWaveletFlow with a HCC prior on the Yuuki simulations

model = "waveletflow"
stepsPerResolution = [2, 2, 4, 8, 16, 16, 16, 16, 16, 16, 16]
stepsPerResolution_L = [3] * 8
nLevels = 6     # Number of DWT levels (log_2(nx))
kernel = 3
baseLevel = 1    # Base level for the DWT, baseLevel=1 means that the first DWT level 2x2 pixels
partialLevel = -1 
hiddenChannels = 256
n_res_blocks = 3
actNormScale = 1.0
imShape = [2, 64, 64]  #this is the shape of the training data 
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
unnormalize_prior = False

#datset configs 
dataset = 'My_lmdb'   #replace with 'My_lmdb' with your dataset
channels_to_get = ['kappa', 'cib']
noise_dict = {
    'kappa': [0.025, None]
}
data_shape = (5, 64, 64)   #this is the shape of the data in the dataset
dataset_path = '/sdf/group/kipac/users/mati/yuuki_sim_train_64x64/'
val_dataset_path = '/sdf/group/kipac/users/mati/yuuki_sim_val_64x64/'
sample_batch_size = 512
normalize = [True] * 10
norm_type = ['min_max']*10

#powerspectra and normalization
std_path = 'norm_stds/64x64_final_mean_stats_all_levels_noise_0.025.json'
ps_path = 'ps/64x64_kappa_noise_0.025_yuuki_kap_cib/dwtlevel'

#output paths
saveDir = '/sdf/group/kipac/users/mati/best_model_64_noised_kappa_rqs/'
plotSaveDir = 'plots_64_noised_kappa_rqs/' 