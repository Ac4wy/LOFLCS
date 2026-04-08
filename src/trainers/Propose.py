from src.trainers.base import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
from src.trainers.Exp3Scheduler import Exp3Scheduler
import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader

class ProposeTrainer(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power, g_N0):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        # self.optimizer = MyAdam(model.parameters(), lr=options['lr'])
        self.optimizer = GD(model.parameters(), lr=options['lr'])
        super(ProposeTrainer, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, g_N0, model
                                             , self.optimizer)
       

        self.client_utilities = {}
        self.client_data_qualities = {}

        self.w_grad = float(options.get("w_grad_init", 0.2))
        self.w_data = 1.0 - self.w_grad

        self.d_w_n = float(options.get("d_w_n", 0.4))
        self.d_w_h = float(options.get("d_w_h", 0.3))
        self.d_w_k = float(options.get("d_w_k", 0.3))
        s = self.d_w_n + self.d_w_h + self.d_w_k
        if s <= 0: self.d_w_n = self.d_w_h = self.d_w_k = 1/3
        else:
            self.d_w_n /= s; self.d_w_h /= s; self.d_w_k /= s

        self.sample_mode = options.get("sample_mode", "top2k_rand")
        self.softmax_temp = float(options.get("softmax_temp", 1.0))
        self.ref_grad = None
        self.ref_beta = float(options.get("ref_beta", 0.8))  

        self.latest_global_model = self.get_model_parameters()

        self.cpu_frequency = np.array(cpu_frequency, dtype=float); 
        if self.cpu_frequency.ndim==0: self.cpu_frequency=np.full(len(self.clients), float(self.cpu_frequency))
        self.Bw = np.array(B, dtype=float)
        self.Ptx = np.array(transmit_power, dtype=float); 
        if self.Ptx.ndim==0: self.Ptx=np.full(len(self.clients), float(self.Ptx))
        self.gN0 = np.array(g_N0, dtype=float); 
        if self.gN0.ndim==0: self.gN0=np.full(len(self.clients), float(self.gN0))

       
        self.model_bytes = sum(p.numel() for p in self.model.parameters()) * 4
        self.cycles_per_sample = float(options.get("cycles_per_sample", 2e6))
        self.kappa = float(options.get("kappa", 1e-28))
        self.local_epochs = int(options.get("local_epoch", 1))
        self.uplink_compress = float(options.get("uplink_compress", 1.0))
                      
        self.V = float(options.get("V_tradeoff", 1))
        self.Qe, self.Qd = 0.0, 0.0

        self.energy_budget_abs = options.get("energy_budget", None)  
        self.delay_budget_abs  = options.get("delay_budget",  None)
        self.adaptive_budget   = bool(options.get("adaptive_budget", True))
        self.warmup_rounds     = int(options.get("warmup_rounds", 5))
        self.energy_budget_factor = float(options.get("energy_budget_factor", 0.9))  
        self.delay_budget_factor  = float(options.get("delay_budget_factor", 0.9))
        self.ma_beta = float(options.get("ma_beta", 0.9))  
        self.baseline_E_ma = None
        self.baseline_D_ma = None

      
    def train(self):
        print('>>> Select {} clients per round \n'.format(int(self.per_round_c_fraction * self.clients_num)))
        for round_i in range(self.num_round):
            self.test_latest_model_on_testdata(round_i)

            stage = round_i / max(1, self.num_round - 1)
            self.w_grad = 0.2 + 0.5 * stage
            self.w_data = 1.0 - self.w_grad

            util_scores, prob = self.compute_sampling_scores_and_probs(round_i)  
            Ei, Di = self.estimate_energy_delay_per_client()

            Be = self.energy_budget_abs
            Bd = self.delay_budget_abs
            if self.adaptive_budget:
                
                if (self.baseline_E_ma is None) or (self.baseline_D_ma is None):
                    self.baseline_E_ma = np.mean(Ei)
                    self.baseline_D_ma = np.percentile(Di, 90) 
                Be = Be if Be is not None else self.energy_budget_factor * self.baseline_E_ma
                Bd = Bd if Bd is not None else self.delay_budget_factor  * self.baseline_D_ma

            sel_idx = self.select_clients_dpp_set_aware(
                util=util_scores, Ei=Ei, Di=Di, K=int(self.per_round_c_fraction * self.clients_num),
                Be=Be, Bd=Bd
            )
            selected_clients = [self.clients[i] for i in sel_idx]
            for c in selected_clients:
                self.metrics.participation_history[c.id].append(1)
                c.increment_participation()

            
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)
           
            self.metrics.update_cost(round_i, self.cost.delay_Sum, self.cost.energy_Sum)
            self.metrics.extend_communication_stats(round_i, stats)
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)

            
            self.update_ref_grad_with(selected_clients)

            E_round = float(np.sum(Ei[sel_idx]))
            D_round = float(np.max(Di[sel_idx])) if len(sel_idx) > 0 else 0.0

            
            if Be is not None:
                self.Qe = max(0.0, self.Qe + E_round - float(Be))
            if Bd is not None:
                self.Qd = max(0.0, self.Qd + D_round - float(Bd))

           
            if self.baseline_E_ma is None:
                self.baseline_E_ma = E_round
            else:
                self.baseline_E_ma = self.ma_beta * self.baseline_E_ma + (1 - self.ma_beta) * E_round
            if self.baseline_D_ma is None:
                self.baseline_D_ma = D_round
            else:
                self.baseline_D_ma = self.ma_beta * self.baseline_D_ma + (1 - self.ma_beta) * D_round

        self.test_latest_model_on_testdata(self.num_round)
        self.metrics.write()

    def compute_sampling_scores_and_probs(self, round_i, eps=1e-10):
        data_scores_dict = self.compute_client_data_scores()
        data_scores = np.array([data_scores_dict[i]["final_score"] for i in range(len(self.clients))], dtype=float)

        grad_align = []
        for i, client in enumerate(self.clients):
            if self.ref_grad is None:
                align = 0.5
            else:
                align = float(client.compute_CQ(self.ref_grad, round_i))
            grad_align.append(align)
        grad_align = np.array(grad_align, dtype=float)

        def _norm01(arr):
            arr = np.asarray(arr, dtype=float)
            a, b = np.min(arr), np.max(arr)
            if b - a < 1e-12: return np.full_like(arr, 0.5)
            return (arr - a) / (b - a + eps)

        data_norm = _norm01(data_scores)
        grad_norm = _norm01(grad_align)
        util = self.w_grad * grad_norm + self.w_data * data_norm 

        t = max(self.softmax_temp, eps)
        logits = util / t
        logits -= np.max(logits)
        prob = np.exp(logits); prob /= (np.sum(prob) + eps)

        self.client_utilities = {
            "GradAlign_norm": grad_norm.tolist(),
            "Data_norm": data_norm.tolist(),
            "Combined(Util)": util.tolist(),
            "prob": prob.tolist(),
        }
        self.client_data_qualities = data_scores_dict
        return util, prob

           
    def estimate_energy_delay_per_client(self, eps=1e-12):
        n = len(self.clients)
        n_i = np.array([len(self.clients_label[i]) for i in range(n)], dtype=float)

        cycles = self.cycles_per_sample * n_i * max(1, self.local_epochs)
        f = self.cpu_frequency
        T_comp = cycles / (f + eps)
        E_comp = self.kappa * cycles * (f ** 2)

        payload_bits = float(self.model_bytes * self.uplink_compress * 8.0)
        snr = self.Ptx * self.gN0
        rate = self.Bw * np.log2(1.0 + np.maximum(snr, 1e-9))
        T_comm = payload_bits / (rate + eps)
        E_comm = self.Ptx * T_comm

        delay = (T_comp + T_comm).astype(float)  
        energy = (E_comp + E_comm).astype(float)  
        return energy, delay

   
    def select_clients_dpp_set_aware(self, util, Ei, Di, K, Be=None, Bd=None):
       
        N = len(self.clients)
        K = min(K, N)
        candidates = list(range(N))
        S = []
        dmax = 0.0 

       
        if Bd is not None:
            fast_pool = [i for i in candidates if Di[i] <= Bd]
            if len(fast_pool) >= max(1, int(0.5*K)):
                candidates = list(set(fast_pool) | set(np.argsort(Di)[:max(K*2, 1)]))

        for _ in range(K):
            best_j, best_gain = None, -1e30
            
            eps = 0.05
            if np.random.rand() < eps and len(candidates) > 0:
                j = np.random.choice(candidates)
                S.append(j); dmax = max(dmax, float(Di[j])); candidates.remove(j)
                continue
            for j in candidates:
                delay_inc = max(0.0, float(Di[j]) - dmax)             
                gain = self.V * float(util[j]) - (self.Qe * float(Ei[j]) + self.Qd * delay_inc)
                if gain > best_gain:
                    best_gain, best_j = gain, j
            if best_j is None:
                break
            S.append(best_j)
            dmax = max(dmax, float(Di[best_j]))
            candidates.remove(best_j)

        if len(S) > 0:
            topk = list(np.argsort(-util)[:min(int(K*0.4), N)])
            pool = list(set(S) | set(topk))
            if len(pool) >= len(S):
                S = list(np.random.choice(pool, len(S), replace=False))


        return S
    def _norm01_arr(self, x, eps=1e-12):
        x = np.asarray(x, dtype=float)
        lo, hi = np.min(x), np.max(x)
        if hi - lo < 1e-12: return np.full_like(x, 0.5)
        return (x - lo) / (hi - lo + eps)

  
    def update_ref_grad_with(self, selected_clients):
        if len(selected_clients) == 0 or getattr(selected_clients[0], "last_gradient", None) is None:
            return
        keys = list(selected_clients[0].last_gradient.keys())
        avg_now = {k: torch.mean(torch.stack([c.last_gradient[k] for c in selected_clients]), dim=0) for k in keys}
        if self.ref_grad is None:
            self.ref_grad = {k: v.detach().clone() for k, v in avg_now.items()}
        else:
            for k in keys:
                self.ref_grad[k] = self.ref_beta * self.ref_grad[k] + (1 - self.ref_beta) * avg_now[k]

    
    def compute_client_data_scores(self, tau_kl=0.8, eps=1e-10):
        num_classes = max(10, int(max(self.dataset.trainLabel)) + 1 if len(self.dataset.trainLabel) > 0 else 10)
        g = np.array(self.get_global_label_distribution(), dtype=float) + eps
        g = g / g.sum()
        all_n = [len(self.clients_label[i]) for i in range(len(self.clients))]
        if len(all_n) == 0: max_log_n = 1.0
        else:
            lo, hi = np.percentile(all_n, 5), np.percentile(all_n, 95)
            all_n_clip = [min(max(n, lo), hi) for n in all_n]
            max_log_n = np.log(1 + max(all_n_clip))

        results = {}
        for i in range(len(self.clients)):
            idxs = self.clients_label[i]
            labels = np.array(self.dataset.trainLabel[idxs], dtype=int)
            n_i = len(labels)
            if n_i == 0:
                results[i] = {"n_score": 0.0, "H_score": 0.0, "K_score": 0.0, "final_score": 0.0}
                continue
            n_i_clip = min(max(n_i, 1), np.percentile(all_n, 95)) if len(all_n) > 0 else n_i
            n_score = float(np.log(1 + n_i_clip) / (max_log_n + 1e-10))
            counts = np.bincount(labels, minlength=num_classes).astype(float)
            p = counts / (n_i + 1e-10)
            H = -np.sum(p * np.log(p + 1e-10))
            H_score = float(H / (np.log(num_classes) + 1e-10))
            Dkl = float(np.sum(p * np.log((p + 1e-10) / (g + 1e-10))))
            K_score = float(np.exp(-tau_kl * Dkl))
            final = self.d_w_n * n_score + self.d_w_h * H_score + self.d_w_k * K_score
            results[i] = {"n_score": n_score, "H_score": H_score, "K_score": K_score, "final_score": final}
        self.client_data_qualities = results
        return results

    
    def get_label_distribution(self):
        label_distributions = []
        for i in range(len(self.clients)):
            labels = self.dataset.trainLabel[self.clients_label[i]]
            counts = [labels.tolist().count(j) for j in range(10)]
            total = sum(counts) if sum(counts) > 0 else 1
            freqs = [c / total for c in counts]
            label_distributions.append(freqs)
        return label_distributions

    def get_global_label_distribution(self):
        all_labels = []
        for client_label in self.clients_label:
            all_labels.extend(self.dataset.trainLabel[client_label])
        total = len(all_labels) if len(all_labels) > 0 else 1
        counts = [all_labels.count(j) for j in range(10)]
        return [c / total for c in counts]
 