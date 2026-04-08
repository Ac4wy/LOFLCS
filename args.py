import argparse
import torch
from getdata import GetDataSet
from src.utils.tools import dirichlet_split_noniid
from src.utils.paras_generate import paraGeneration
import importlib
from src.utils.torch_utils import setup_seed
from src.utils.tools import plot_client_class_categories
# GLOBAL PARAMETERS
DATASETS = ['mnist',  'cifar10']
TRAINERS = {
            'Propose': 'ProposeTrainer',
            }
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

OPTIMIZERS = TRAINERS.keys()
def input_options():
    parser = argparse.ArgumentParser()
    # iid
    parser.add_argument( '-is_iid', type=bool, default=True, help='data distribution is iid.')
    # parser.add_argument( '--dataset_name', type=str, default='mnist_dir_', help='name of dataset.')
    parser.add_argument( '--dataset_name', type=str, default='cifar10', help='name of dataset.')
    # parser.add_argument('--model_name', type=str, default='mnist_cnn', help='the model to train')
    
    parser.add_argument('--model_name', type=str, default='cifar10_alexnet', help='the model to train')
    parser.add_argument('--gpu', type=bool, default=True, help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('--round_num', type=int, default=501, help='number of round in comm')
    parser.add_argument( '--num_of_clients', type=int, default=100, help='numer of the clients')
    parser.add_argument( '--c_fraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('--local_epoch', type=int, default=5, help='local train epoch')
    parser.add_argument( '--batch_size', type=int, default=32, help='local train batch size')
    parser.add_argument( "--lr", type=float, default=0.1, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument( '--transmit_power', type=int, default=1, help='transmitpower')
    parser.add_argument( '--gn0', type=int, default=1, help='gno')
    parser.add_argument('--seed', help='seed for randomness;', type=int, default=0)
    parser.add_argument( '--weight_decay', help='weight_decay;', type=int, default=1)
    parser.add_argument( '--algorithm', help='algorithm;', choices=OPTIMIZERS, type=str, default='Propose')
    parser.add_argument( '--dirichlet', help='Dirichlet;', type=float, default=0.1)
    parser.add_argument('--opti', help='Dirichlet;', type=str, default='sgd')
      # === New: Fed-HYPE-Lite (greedy-only) hyper-params ===
    parser.add_argument('--tau_max', type=float, default=5.0, help='max per-round latency budget (seconds)')
    parser.add_argument('--E_max', type=float, default=3e2, help='per-round energy budget (arbitrary units)')
    parser.add_argument('--alpha_gim', type=float, default=0.5, help='weight inside GIM: loss_pre vs delta_w')
    parser.add_argument('--w_gim', type=float, default=1.0, help='weight of gradient/model value in objective')
    parser.add_argument('--lambda_E', type=float, default=0.001, help='energy penalty weight in objective')
    parser.add_argument('--coverage', type=str, default='log', choices=['log', 'sqrt'],
                        help='concave coverage function for class counts')
    parser.add_argument('--max_per_round', type=int, default=None,
                        help='optional hard cap of clients per round; default uses c_fraction')
    args = parser.parse_args()
    options = args.__dict__
    # dataset = GetDataSet(options['dataset_name'][:5]) # 拿到数据集 分配完再导入
    dataset = GetDataSet(options['dataset_name'][:]) # 拿到数据集 分配完再导入
    client_label, result = dirichlet_split_noniid(dataset.trainLabel, options['dirichlet'], options['num_of_clients'])
 
    plot_client_class_categories(client_label[:20], 20, 10, dataset.trainLabel)

    cpu_frequency, B, transmit_power, g_N0 = paraGeneration(options)
    setup_seed(options['seed'])
    trainer_path = 'src.trainers.%s' % options['algorithm']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algorithm']])
    trainer = trainer_class(options, dataset, client_label, cpu_frequency, B, transmit_power, g_N0)
    return trainer


