model = "waveletflow"
stepsPerResolution = [2, 2, 4, 8, 16, 16, 16, 16, 16, 16, 16]
stepsPerResolution_L = [3] * 8
normalize = [False] * 10
nLevels = 8
kernel = 3
baseLevel = 1
partialLevel = -1
hiddenChannels = 256
n_res_blocks = 3
actNormScale = 1.0
imShape = [2, 256, 256]
perm = "invconv"
coupling = "checker"
y_classes = None
y_learn_top = False
y_condition = False
y_weight = None
conditional = True
LU = True

#prior
gauss_priors = [1,2,3,4,5,6,7] #set this for HCC prior. For WN prior, add all the layers to this list
priorType = 'CC' #Choose 'CC' or 'C'
unnormalize_prior = False

#datset configs 
dataset = 'My_lmdb'   #replace with 'My_lmdb' with your dataset
channels_to_get = ['kappa', 'cib']
noise_dict = {
    'kappa': 0.025
}
data_shape = (5, 256, 256)   #this is the shape of the data in the dataset
dataset_path = '/sdf/group/kipac/users/mati/yukki_sim_train_256x256/'
val_dataset_path = '/sdf/group/kipac/users/mati/yukki_sim_train_256x256/'
sample_batch_size = 128

#powerspectra and normalization
std_path = 'norm_stds/256x256_final_mean_stats_all_levels_noise_0.025.json'
ps_path = 'ps/256x256_noised_0.025_sim_kapcib_dwtlevel'

#output paths
saveDir = '/sdf/group/kipac/users/mati/HCC_prior_best_model_256/'
plotSaveDir = 'plots_HCC_prior_best_model_256/' 