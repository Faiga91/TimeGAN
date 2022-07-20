
# %%
from IPython import get_ipython

IPYTHON_INSTANCE = get_ipython()
IPYTHON_INSTANCE.run_line_magic('load_ext','autoreload')
IPYTHON_INSTANCE.run_line_magic('autoreload','2')
# %%
## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os 
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath('__file__'))[:-9] + 'code/'
sys.path.append(os.path.dirname(SCRIPT_DIR))

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

# %% [markdown]
# ## Data Loading
# 
# Load original dataset and preprocess the loaded data.
# 
# - data_name: stock, energy, or sine
# - seq_len: sequence length of the time-series data

# %%
## Data loading
data_name = 'stock'
seq_len = 24

if data_name in ['stock', 'energy']:
  ori_data = real_data_loading(data_name, seq_len)
elif data_name == 'sine':
  # Set number of samples and its dimensions
  no, dim = 10000, 5
  ori_data = sine_data_generation(no, seq_len, dim)
    
print(data_name + ' dataset is ready.')

# %% [markdown]
# ## Set network parameters
# 
# TimeGAN network parameters should be optimized for different datasets.
# 
# - module: gru, lstm, or lstmLN
# - hidden_dim: hidden dimensions
# - num_layer: number of layers
# - iteration: number of training iterations
# - batch_size: the number of samples in each batch

# %%
## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 10000
parameters['batch_size'] = 128

# %% [markdown]
# ## Run TimeGAN for synthetic time-series data generation
# 
# TimeGAN uses the original data and network parameters to return the generated synthetic data.

# %%
# Run TimeGAN
generated_data = timegan(ori_data, parameters)   
print('Finish Synthetic Data Generation')

# %% [markdown]
# ## Evaluate the generated data
# 
# ### 1. Discriminative score
# 
# To evaluate the classification accuracy between original and synthetic data using post-hoc RNN network. The output is |classification accuracy - 0.5|.
# 
# - metric_iteration: the number of iterations for metric computation.

# %%
metric_iteration = 5

discriminative_score = list()
for _ in range(metric_iteration):
  temp_disc = discriminative_score_metrics(ori_data, generated_data)
  discriminative_score.append(temp_disc)

print('Discriminative score: ' + str(np.round(np.mean(discriminative_score), 4)))

# %% [markdown]
# ## Evaluate the generated data
# 
# ### 2. Predictive score
# 
# To evaluate the prediction performance on train on synthetic, test on real setting. More specifically, we use Post-hoc RNN architecture to predict one-step ahead and report the performance in terms of MAE.

# %%
predictive_score = list()
for tt in range(metric_iteration):
  temp_pred = predictive_score_metrics(ori_data, generated_data)
  predictive_score.append(temp_pred)   
    
print('Predictive score: ' + str(np.round(np.mean(predictive_score), 4)))

# %% [markdown]
# ## Evaluate the generated data
# 
# ### 3. Visualization
# 
# We visualize the original and synthetic data distributions using PCA and tSNE analysis.

# %%
visualization(ori_data, generated_data, 'pca')
visualization(ori_data, generated_data, 'tsne')

# %%



