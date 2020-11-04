import torch
import numpy as np

def reinit_valid_set(dl_train_list, number_cluster=None):
    clusters = []
    return None

from easydict import EasyDict
import os

cfg = EasyDict()
cfg.sop_dataset_path = '/media/natalia/5cd27ae5-3a27-4289-a36e-42ebe238d3da/Datasets/Stanford_Online_Products/'
cfg.inshop_dataset_path = '/media/natalia/5cd27ae5-3a27-4289-a36e-42ebe238d3da/Datasets/In_shop Clothes Retrieval Benchmark/'
cfg.cub_dataset_path = '/kaggle/input/cub200/CUB_200_2011'
cfg.cars_dataset_path = '/kaggle/input/cars196'
cfg.pretrained_pytorch_models = '/kaggle/input/pretrained-pytorch-models'

cfg.device = 'cuda:0'
cfg.random_seed = 0
cfg.lr = 5e-5
cfg.embedding_dim = 128
cfg.masks_lr = 1e-5

# OnlineProducts params
cfg.sop_train_classes = range(0, 11318)
cfg.sop_test_classes = range(11318, 22634)
cfg.sop_nb_train_data = 59551

# InShop params
cfg.inshop_train_classes = range(0, 3997)
cfg.inshop_test_classes = range(0, 3985)
cfg.inshop_nb_train_data = 25882

# VehicleId params
cfg.vid_train_classes = range(0, 13164)
cfg.vid_test_classes = {'small': range(13164, 13964), 'medium': range(13164, 14764), 'large': range(13164, 15564)}
cfg.vid_nb_train_data = 113346

# CARS196 params
cfg.cars_train_classes = range(1, 99)
cfg.cars_test_classes = range(99, 197)
cfg.cars_nb_train_data = 8054

# CUB200 params
cfg.cub_train_classes = range(1, 101)
cfg.cub_test_classes = range(101, 201)
cfg.cub_nb_train_data = 5864

cfg.epochs = 300
cfg.fine_tune_epoch = 300  # 190
cfg.recluster_epoch_freq = 2

cfg.load_saved_model_for_test = True
cfg.save_clusters = True
cfg.save_model = True
cfg.log_to_mlflow = True
cfg.log_to_tensorboard = True
cfg.use_amp = True
cfg.to_make_density_plots = True
cfg.clustering_by_embeddings = False
cfg.use_normalized_emb_while_clustering = True

cfg.continue_training_from_epoch = False
cfg.checkpoint_from_epoch = 99
cfg.clusters_from_epoch = 98


# to eval model with loaded checkpoint or to check model performance before training
cfg.compute_metrics_before_training = False
cfg.evalaute_on_train_data = False
cfg.lambda1 = 5e-4  # masks l1 regularization
cfg.lambda2 = 5e-3

# adaptive margins params
cfg.use_adaptive_margins = False
cfg.adaptive_margin_start_epoch = 100
cfg.use_clusters_own_not_null_parts = False #True
cfg.not_null_part = 0.5 # None
cfg.margin_type = 'const' # ['linear', 'ema', 'const']
cfg.margin_lambda = 0.999

# exp-linear loss params
cfg.use_exp_linear_loss = False
cfg.exp_linear_loss_start_epoch = 100

# fast_valid params
cfg.fast_valid = False
cfg.batch_size_fast_valid = 256

cfg.finetune_masks = False

cfg.available_datasets = [
    'OnlineProducts',
    'InShopClothes',
    'VehicleId',
    'CUB200',
    'CARS196'
]
cfg.cur_dataset = 'InShopClothes'
cfg.batch_size = 80 if cfg.cur_dataset in ['OnlineProducts', 'InShopClothes'] else 128

cfg.use_gem = True
cfg.use_lr_scheduler = True
cfg.use_mean_beta_at_finetune = False
cfg.use_cls_loss = False

if cfg.cur_dataset == 'OnlineProducts':
    cfg.train_classes = cfg.sop_train_classes
    cfg.k_list = [1, 10, 100, 1000]
elif cfg.cur_dataset == 'InShopClothes':
    cfg.train_classes = cfg.inshop_train_classes
    cfg.k_list = [1, 10, 20, 30, 50]
elif cfg.cur_dataset == 'VehicleId':
    cfg.train_classes = cfg.vid_train_classes
    cfg.k_list = [1, 5]
elif cfg.cur_dataset == 'CUB200':
    cfg.train_classes = cfg.cub_train_classes
    cfg.k_list = [1, 2, 4, 8]
elif cfg.cur_dataset == 'CARS196':
    cfg.train_classes = cfg.cars_train_classes
    cfg.k_list = [1, 2, 4, 8]
else:
    raise Exception


# if cfg.cur_dataset in ['CUB200', 'CARS196']:
#     cfg.nb_clusters = 4
# else:
#     cfg.nb_clusters = 8

cfg.nb_clusters = 8

cfg.own_learner = True
cfg.transformer_unit = True
cfg.transformer_unit_on_learner_train = False

cfg.ft_epoch = 60 if cfg.cur_dataset in ['CUB200', 'CARS196'] else 100

cfg.ft_epoch = 60 if cfg.cur_dataset is 'CUB200' or cfg.cur_dataset is 'CARS196' else 100
cfg.run_dir = f'/media/natalia/5cd27ae5-3a27-4289-a36e-42ebe238d3da/Divide-and-conquer/{cfg.cur_dataset}/nb_clusters_{cfg.nb_clusters}_without_norm_with_MHA_on_finetune_trainable_beta{cfg.nb_clusters+1}_margin_{cfg.margin_type}/'
if not os.path.exists(cfg.run_dir):
    os.makedirs(cfg.run_dir)
cfg.input_dir = '/media/natalia/5cd27ae5-3a27-4289-a36e-42ebe238d3da/Divide-and-conquer/InShopClothes/nb_clusters_8_without_norm_with_own_fc_trainable_beta9_adaptive_margin_ema3/' #cfg.run_dir
cfg.load_all_metrics = False # True if continue training
cfg.all_metrics_dir = cfg.input_dir+'all_metrics'  # f'/kaggle/input/checkpoints/all_metrics_{cfg.checkpoint_from_epoch}'

cfg.checkpoints_dir = cfg.run_dir
cfg.tensorboard_dir = cfg.run_dir
cfg.losses_dir = cfg.run_dir
cfg.metrics_dir = cfg.run_dir
cfg.density_plots_dir = cfg.run_dir



class BatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, dataset, batch_size, m=4):
        self.m = m
        self.dataset = dataset
        self.batch_size = batch_size

        self.prod_labels = np.array(dataset.labels)
        self.unique_prod_labels = list(set(self.prod_labels))

        self.prod_label_indices_within_cur_label = self.get_all_prod_labels_indices_with_cur_label()
        self.nb_classes_to_pick = self.batch_size // self.m

        self.indices_real_from_class = self.get_all_prod_labels_indices_with_cur_label()
        self.indices_taken_from_class = np.zeros(len(self.unique_prod_labels))

    def get_all_prod_labels_indices_with_cur_label(self):
        out = []
        for c in self.unique_prod_labels:
            prod_label_indices = np.where(self.prod_labels == c)[0]
            np.random.shuffle(prod_label_indices)
            out.append(prod_label_indices)
        return out

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        if cfg.fast_valid:
            re_init(self)
        for _ in range(len(self.dataset) // self.batch_size):
            if len(self.unique_prod_labels) > self.nb_classes_to_pick:
                replace = False
            else:
                replace = True
            chosen_classes = np.random.choice(
                len(self.unique_prod_labels), self.nb_classes_to_pick,
                replace=replace)
            out = []
            for cl in chosen_classes:
                if self.indices_taken_from_class[cl] + self.m \
                        < len(self.indices_real_from_class[cl]):
                    st = int(self.indices_taken_from_class[cl])
                    chosen_indices = self.indices_real_from_class[cl][st:st + self.m]
                else:
                    chosen_indices = np.random.choice(self.indices_real_from_class[cl],
                                                      self.m,
                                                      replace=len(self.indices_real_from_class[cl]) < self.m)
                out.extend(chosen_indices)
                self.indices_taken_from_class[cl] += self.m

                if self.indices_taken_from_class[cl] + self.m \
                        > len(self.indices_real_from_class[cl]):
                    np.random.shuffle(self.indices_real_from_class[cl])
                    self.indices_taken_from_class[cl] = 0
            yield out
