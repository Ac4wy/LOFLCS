from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch
import copy
criterion = F.cross_entropy
mse_loss = nn.MSELoss()
EPS = 1e-12

class Client():
    def __init__(self, options, id, attr, local_dataset, model, optimizer, ):
        self.options = options
        self.id = id
        self.local_dataset = local_dataset
        self.model = model
        self.gpu = options['gpu']
        self.optimizer = optimizer
        self.flops, self.params_num, self.model_bytes = get_model_complexity_info(self.model,(3, 32, 32), gpu=options['gpu'])#mnist
            # get_model_complexity_info(self.model, (3, 32, 32), gpu=options['gpu']) (1, 28, 28)
       
        self.attr_dict = attr.get_client_attr(self.id)
    
        self.cc = self.attr_dict['cpu_frequency'] * 1e9 / 10000.0   # 样本/秒
        B_hz = self.attr_dict['B'] * 1e6               # MHz -> Hz
        snr = self.attr_dict['transmit_power'] * self.attr_dict['g_N0']
 
        self.selected_in_round = False
        self.prev_gradient = None
        self.last_gradient = None

        self.attr_dict = attr.get_client_attr(self.id)
        self.last_gradient = None
        self.participation_count = 0  
        self.data_quality = None  
    

        
    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def get_train_loader(self, options):
        """动态创建并返回数据加载器"""
        return DataLoader(
            self.local_dataset,
            batch_size=options['batch_size'],
            shuffle=True  # 通常训练时需要shuffle
        )
    def local_train(self, ):
        bytes_w = self.model_bytes
        begin_time = time.time()
        local_model_paras, dict = self.local_update(self.local_dataset, self.options, )
        end_time = time.time()
        bytes_r = self.model_bytes
        stats = {'id': self.id, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats

    def local_update(self, local_dataset, options, ):
        localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        # print(self.optimizer.param_groups[0]['lr'])
        train_loss = train_acc = train_total = 0
        gradients = [] # 存储梯度
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu:
                    X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                #收集梯度
                if epoch == options['local_epoch']-1: #对于最后一轮梯度
                    grad_dict = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_dict[name] = param.grad.clone()
                    gradients.append(grad_dict)
                # print(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
        
        #计算平均梯度
        avg_grad = {}
        if gradients:
            for key in gradients[0].keys():
                avg_grad[key] = torch.stack([g[key] for g in gradients]).mean(dim=0)
        
                # === 新增：计算梯度创新度（与上一轮该客户端梯度差的范数） ===
        if self.last_gradient is not None and avg_grad:
            try:
                prev_vec = torch.cat([v.flatten() for v in self.last_gradient.values()])
                curr_vec = torch.cat([v.flatten() for v in avg_grad.values()])
                self.grad_innovation_norm = torch.norm(curr_vec - prev_vec).item()
            except Exception:
                self.grad_innovation_norm = 0.0
        else:
            self.grad_innovation_norm = 0.0

        # 滚动保存梯度
        self.prev_gradient = self.last_gradient
        self.last_gradient = avg_grad 
        local_model_paras = self.get_model_parameters()
        comp = self.options['local_epoch'] * train_total * self.flops
        return_dict = {"id": self.id,
                       "comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total,
                       "gradient": avg_grad}

        return local_model_paras, return_dict

    def getLocalEngery(self):
        localEngery = (10 ** -26) * (self.attr_dict['cpu_frequency'] * 1000000000) ** 2 * 10000 * len(self.local_dataset)
        return localEngery

    def getUploadEngery(self):
        uploadEngery = self.attr_dict['transmit_power'] * self.getUploadDelay()
        return uploadEngery

    def getLocalDelay(self):
        localDelay = 10000 * len(self.local_dataset) * self.options['local_epoch'] / (self.attr_dict['cpu_frequency'] * 1000000000)
        return localDelay

    def getUploadDelay(self):
        R_K = self.attr_dict['B'] * 1000000 * np.log2(1 + self.attr_dict['transmit_power'] * self.attr_dict['g_N0']) # 1M bit / s / self.B
        uploadDelay = 21.98/ (R_K / 8 / 1024 / 1024) # 100KB 0.1M  # 1S
        #21.98MB 6.35
        return uploadDelay

    def getSumEngery(self):
        # 返回上传能量和本地能量的总和
        return self.getUploadEngery() + self.getLocalEngery()

    def getSumDelay(self):
        return self.getUploadDelay() + self.getLocalDelay()
    def compute_CQ(self, global_grad, round_i):
        if self.last_gradient is None or global_grad is None:
            return 0.0

        try:
            local_grad_vec = torch.cat([
                torch.flatten(v) for v in self.last_gradient.values()
            ])
            global_grad_vec = torch.cat([
                torch.flatten(v) for v in global_grad.values()
            ])
        except Exception as e:
            print(f"[CQ Error] Gradient flatten failed: {e}")
            return 0.0

        # 计算余弦相似度（方向差异）
        cos_sim = F.cosine_similarity(local_grad_vec, global_grad_vec, dim=0)
        
        local_norm = torch.norm(local_grad_vec)
        global_norm = torch.norm(global_grad_vec) + 1e-10
        norm_ratio = abs(local_norm - global_norm) / global_norm


        stage = min(round_i / self.options['round_num'], 1.0)
        # return ((1 - cos_sim) * (1 + stage * norm_ratio)).item()
        s1 = (1+cos_sim) / 2
        s2 = 1 / (1+ norm_ratio)
        s = s1 * s2
        return s.item()
