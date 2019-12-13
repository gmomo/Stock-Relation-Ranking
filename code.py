import torch
import torch.nn as nn
import numpy as np
import argparse

def generate_ts_data(n_stocks, max_seq_len, feature_dim):
    synth_data = [np.random.rand(max_seq_len, feature_dim) for i in range(n_stocks)]
    synth_data = torch.stack(synth_data)
    return synth_data

def generate_ranking_data(n_stocks, max_seq_len, seq_len):
    data_len = max_seq_len - seq_len
    ranking_labels =  torch.Tensor( 2*np.random.random_sample( (n_stocks, data_len) ) - 1 )
    return ranking_labels

def generate_rel_data(n_stocks, k):
    rel_data = torch.Tensor( np.random.randint(2, size=(n_stocks, n_stocks, k)) )
    return rel_data

def LSTM_Layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)

    def forward(self, x):
        lstm_output = self.lstm_layer(x)
        return lstm_output

def RelEmbed_Layer(nn.Module):
    def __init__(self,model_type,k,type="explicit", lstm_hidden_size=None):
        
        self.weight = torch.Tensor( (1,k) )
        self.bias = torch.Tensor( (1,1) )
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def _compute_mask(rel_data):
        """Compute Mask to multiply with the strength rel data"""
        _sum_rel = torch.sum(rel_data, dim=2) # N X N sum(aji)
        _normalize_rel = torch.sum(_sum_rel, dim=1).unsqueeze(1) # N X 1 to calculate d_j
        _sum_rel[ _sum_rel != 0] = 1 #Place 1 in all places where not zero
        mask = torch.div(_sum_rel, _normalize_rel)

        return mask

    def forward(self, embeddings, rel_data, i):
        #Embeddings shape N X U N = number of stocks, U = hidden_size
        # rel_data N X N X K = (i,j) 
        n,u = embeddings.shape[0], embeddings.shape[1]

        embeddings_dot_product = torch.mm(embeddings, embeddings.transpose(1,0)) #eiT dot Ej N X N
        weight_expand = self.weight.transpose(1,0).unsqueeze(0).repeat(n,1,1) #N X K X 1
        weights = self.leaky_relu ( torch.bmm(embeddings_dot_product, weight_expand).squeeze(2) + self.bias ) # N X N 

        impact_strength = torch.mul(embeddings_dot_product, weights) #G(aij, eit, eij) N X N
        masked_impact_strength = torch.mul(mask * impact_strength) #N * N 

        #Final Tensor ## Can be optimized 
        temp = torch.zeros((1, u))
        return temp


def FC_Layer(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        self.fc_layer = nn.Linear(self.input_size,1)

    def forward(self, x):
        fc_output = self.fc_layer(x)
        return fc_output


def loss_function(r_truth, r_pred, alpha):
    mse_loss = nn.MSELoss()
    reg_loss = mse_loss(r_pred, r_truth)

    r_truth_mat = r_truth.unsqueeze(1) - r_truth
    r_pred_mat = r_pred.unsqueeze(1) - r_pred

    pair_rank_loss = max(torch.zeroes(r_truth.shape[0]), -torch.mul(r_truth_mat, r_pred_mat) ).sum()
    return reg_loss + alpha*pair_rank_loss

def main(args):

    stock_data = generate_ts_data(args.n_stocks, args.max_seq_len, args.feature_dim)
    print(stock_data.shape)
    relation_data = generate_rel_data(args.n_stocks, args.k)
    ranking_label = generate_ranking_data(args.n_stocks, args.max_seq_len, args.seq_len)





if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_stocks',type=int, default=5,  help='Number of stocks') 
    parser.add_argument('--max_seq_len', type=int, default=10, help='Sequence Length of Data to be generated')
    parser.add_argument('--seq_len', type=int, default=5, help='Sequence Length of Data to be fed to LSTM')
    parser.add_argument('--feature_dim', type=int, default=16, help='Feature Dimension of Data')
    parser.add_argument('--k', type=int, default=30, help='Size of Multi-hot Vector of Relational Attributes')
    parser.add_argument('--lstm_hidden_size', type=int, default=64, help='Hidden Size of LSTM')


    args = parser.parse_args()
    main(args)