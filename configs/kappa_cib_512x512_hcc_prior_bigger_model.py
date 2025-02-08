#this is an example config file for training a HybridPriorWaveletFlow with a HCC prior on the Yuuki simulations

model = "waveletflow"
stepsPerResolution = [2, 4, 8, 8, 8, 8, 8, 8, 16, 16, 16]
stepsPerResolution_L = [None] * 10
nLevels = 9    # Number of DWT levels (log_2(nx))
kernel = 3
baseLevel = 1    # Base level for the DWT, baseLevel=1 means that the first DWT level 2x2 pixels
partialLevel = -1 
hiddenChannels = 256
n_res_blocks = 3
actNormScale = 1.0
imShape = [2, 512, 512]  #this is the shape of the training data 
perm = "invconv"
coupling = "checker"  #or "rqs, fully_active_rqs, rqs_per_c or checker"
network = ['ConvNet'] * 10
y_classes = None     
y_learn_top = False
y_condition = False
y_weight = None
conditional = True
LU = True
#prior
gauss_priors = [1,2,3,4,5,6,7,8] #set this for HCC prior. For WN prior, add all the layers to this list
priorType = 'CC' #Choose 'CC' or 'C'
unnormalize_prior = True

#datset configs 
dataset = 'My_lmdb'   #replace with 'My_lmdb' with your dataset
channels_to_get = ['kappa', 'cib']
noise_dict = {
    'kappa' : [1.0, 21600]
}
data_shape = (5, 512, 512)   #this is the shape of the data in the dataset
dataset_path = '/sdf/group/kipac/users/mati/yukki_sim_train_512x512/'
val_dataset_path = '/sdf/group/kipac/users/mati/yukki_sim_val_512x512/'
sample_batch_size = 32
normalize = [False] * 10
norm_type = ['std']*10
double_precision = [None, True, True, True, True, True, False, False, False, False]

#powerspectra and normalization
std_path = 'norm_stds/512x512_final_mean_stats_all_levels_kappa_noise_1.0_noise_freq_21600_kappa_cib.json'
ps_path = 'ps/512x512_kappa_noise_1.0_noise_freq_21600_kappa_cib/dwtlevel'

#output paths
saveDir = '/sdf/group/kipac/users/mati/real_hcc_prior_model_512_noised_kappa_nyquest_noise_bigger_model/'
plotSaveDir = 'plots_hcc_model_512_noised_kappa_nyquest_noise_bigger_model/'