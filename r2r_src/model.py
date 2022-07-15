
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
from utils import Tokenizer
import random

class AuxPredictor(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(AuxPredictor, self).__init__()

        # learnable transformation matrix
        self.linear_tw = nn.Linear(embedding_size, hidden_size, bias=False)
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.sm = nn.Softmax()

    def set_req_grad_false(self):
        for para in self.parameters():
            para.requires_grad = False

    def set_req_grad_true(self):
        for para in self.parameters():
            para.requires_grad = True

    def forward(self, hidden_feature, target_word_feature, mask=None):
        hidden_feature = self.linear_h(hidden_feature) #(bs, Dp)
        target_word_feature = self.linear_tw(target_word_feature) #(bs, K, Dp)
        attack_logit = torch.bmm(target_word_feature, hidden_feature.unsqueeze(1).transpose(1,2)).squeeze(2) #(bs, K)

        attack_logit.masked_fill_(mask.bool(), -float('inf'))

        attack_logit = self.sm(attack_logit)

        return attack_logit


class Attacker(nn.Module):
    def __init__(self, embedding_size, hidden_size, feature_size):
        super(Attacker, self).__init__()
        
        # learnable transformation matrix
        self.linear_v = nn.Linear(feature_size, hidden_size, bias=False)
        self.linear_tw = nn.Linear(embedding_size, hidden_size, bias=False)       
        self.linear_cw = nn.Linear(embedding_size, hidden_size, bias=False)

        self.sm = nn.Softmax()

    def set_req_grad_false(self):
        for para in self.parameters():
            para.requires_grad = False

    def set_req_grad_true(self):
        for para in self.parameters():
            para.requires_grad = True

    def forward(self, feedback_attacker, perm_obs, ctx, visual_feature, \
        target_word_feature, candidate_word_feature, perturb_mask,\
                target_word=None, candidate_word=None, attention=None):

        # get word importance
        visual_feature_h = self.linear_v(visual_feature).unsqueeze(1) # (bs, 1, Dp)
        target_word_feature_h =self.linear_tw(target_word_feature) # (bs, K, Dp)
        word_importance = torch.bmm(target_word_feature_h, visual_feature_h.transpose(1,2)).squeeze(2) # (bs, K)
        word_importance = self.sm(word_importance)

        # get substitution impact matrix
        candidate_word_feature_h = self.linear_cw(candidate_word_feature) # (bs, K, K-1, Dp)
        # get substitution impact matrix for each target word
        substitution_impact = []
        for target_word_ind in range(target_word_feature.size(1)):
            substitution_impact_each_target = torch.bmm(candidate_word_feature_h[:, target_word_ind, :, :], \
                    target_word_feature_h[:, target_word_ind, :].unsqueeze(1).transpose(1,2)).squeeze(2) # (bs, (K-1))

            substitution_impact_each_target = self.sm(substitution_impact_each_target)

            substitution_impact.append(substitution_impact_each_target)

        substitution_impact_s = torch.stack(substitution_impact, dim=1) # (bs, K, (K-1))

        # get attack score (action prediction probability)
        attack_score = word_importance.unsqueeze(2) * substitution_impact_s # (bs, K, (K-1))
        attack_score_final = attack_score.view(attack_score.size(0), attack_score.size(1) * attack_score.size(2))
        attack_score_final.masked_fill_(perturb_mask.bool(), -float('inf'))

        if feedback_attacker == 'sample':
            probs_attacker = F.softmax(attack_score_final, 1)
            c_attacker = torch.distributions.Categorical(probs_attacker)
            a_t_attacker = c_attacker.sample().detach()
            entropy = c_attacker.entropy()
            policy_log_probs = c_attacker.log_prob(a_t_attacker)

        elif feedback_attacker == 'argmax':
            _, a_t_attacker = attack_score_final.max(1)        
            a_t_attacker = a_t_attacker.detach()
            entropy = None
            policy_log_probs = None

        else:
            print(feedback_attacker)
            sys.exit('Invalid feedback option')

        ####make action of attacker####
        # get target word index
        sub_target_word_ind = a_t_attacker / args.maxCWnumber 

        # get candidate word index
        sub_candidate_word_ind = a_t_attacker % args.maxCWnumber

        # substitution operation
        perturbed_ctx = ctx.clone()

        for ins_ind in range(ctx.size(0)):
            if args.pretrain_attacker:
                ins_target_word_pos = perm_obs[ins_ind]['target_word_position']
                ins_candidate_word_pos = perm_obs[ins_ind]['candidate_word_position']
            elif args.adv_train:
                ins_target_word_pos = target_word[ins_ind]['target_word_position']
                ins_candidate_word_pos = candidate_word[ins_ind]['candidate_word_position']

            sub_target_word_pos = int(ins_target_word_pos[sub_target_word_ind[ins_ind]])
            sub_candidate_word_pos = int(ins_candidate_word_pos[sub_target_word_ind[ins_ind]][sub_candidate_word_ind[ins_ind]])
            if sub_target_word_pos < len(ctx[ins_ind])-1 and sub_candidate_word_pos < len(ctx[ins_ind])-1:
                perturbed_ctx[ins_ind][sub_target_word_pos] = ctx[ins_ind][sub_candidate_word_pos]

        return attack_score_final, perturbed_ctx, sub_target_word_ind, entropy, policy_log_probs



class AttnDecoderLSTM_have_attacker(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM_have_attacker, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

        self.attacker = Attacker(args.rnn_dim, args.rnn_dim, \
                feature_size)

        self.aux_predictor = AuxPredictor(args.rnn_dim, args.rnn_dim)

    def set_req_grad_false_for_att(self):
        for k, v in self.named_parameters():
            if k[:8] == 'attacker':
                #print(k)
                v.requires_grad = False

    def set_req_grad_true_for_att(self):
        for k, v in self.named_parameters():
            if k[:8] == 'attacker':
                v.requires_grad = True

    def set_req_grad_false_for_nav(self):
        for k, v in self.named_parameters():
            if k[:8] != 'attacker' and k[:13] != 'aux_predictor':
                #print('nav', k)
                v.requires_grad = False

    def set_req_grad_true_for_nav(self):
        for k, v in self.named_parameters():
            if k[:8] != 'attacker' and k[:13] != 'aux_predictor':
                v.requires_grad = True

    def set_req_grad_false_for_aux(self):
        for k, v in self.named_parameters():
            if k[:13] == 'aux_predictor':
                v.requires_grad = False

    def set_req_grad_true_for_aux(self):
        for k, v in self.named_parameters():
            if k[:13] == 'aux_predictor':
                v.requires_grad = True


    def forward(self, no_perturb,feedback_attacker, perm_obs, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, target_word_enc, candidate_word_enc, \
                perturb_mask, target_word_mask, ctx_mask=None,\
                target_word=None, candidate_word=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)

        if no_perturb:
            perturbed_word_logit = None
            sub_target_word = None
            entropy = None
            policy_log_probs = None
        else:
            perturbed_word_logit, ctx, sub_target_word, \
                entropy, policy_log_probs = self.attacker(feedback_attacker, \
                    perm_obs, ctx, attn_feat, \
                    target_word_enc, candidate_word_enc, perturb_mask, \
                                                target_word, candidate_word)

        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # self-supervised reasoning
        attack_logit = self.aux_predictor(h_tilde, target_word_enc, target_word_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde, attack_logit, sub_target_word, entropy, policy_log_probs

class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def set_req_grad_false(self):
        for para in self.parameters():
            para.requires_grad = False

    def set_req_grad_true(self):
        for para in self.parameters():
            para.requires_grad = True

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)

        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)


        return h_1, c_1, logit, h_tilde

class CriticAttacker(nn.Module):
    def __init__(self):
        super(CriticAttacker, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def set_req_grad_false(self):
        for para in self.parameters():
            para.requires_grad = False

    def set_req_grad_true(self):
        for para in self.parameters():
            para.requires_grad = True

    def forward(self, state):
        return self.state2value(state).squeeze()

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def set_req_grad_false(self):
        for para in self.parameters():
            para.requires_grad = False

    def set_req_grad_true(self):
        for para in self.parameters():
            para.requires_grad = True

    def forward(self, state):
        return self.state2value(state).squeeze()

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def set_req_grad_false(self):
        for para in self.parameters():
            para.requires_grad = False

    def set_req_grad_true(self):
        for para in self.parameters():
            para.requires_grad = True

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float('inf'))

        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


