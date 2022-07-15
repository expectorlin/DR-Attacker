
import sys

# change to your own path
sys.path.insert(0, '/R2R-Aux/build')

import torch

import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker
import pandas as pd

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import warnings
warnings.filterwarnings("ignore")


from tensorboardX import SummaryWriter
#
# prefix = os.environ.get('PREFIX')
# args.name = prefix
# PLOT_DIR = 'tasks/R2R/plots/' + prefix + '/'
# if not os.path.exists(PLOT_DIR):
#     os.makedirs(PLOT_DIR)
#
# SNAPSHOT_DIR = 'tasks/R2R/snapshots/' + prefix + '/'
# if not os.path.exists(SNAPSHOT_DIR):
#     os.makedirs(SNAPSHOT_DIR)


log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES

if args.fast_train:
    name, ext = os.path.splitext(features)
    features = name + "-fast" + ext

feedback_method = args.feedback # teacher or sample
attacker_feedback_method = args.feedback_attacker
#print(feedback_method)

print(args)


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)

    if args.fast_train:
        log_every = 40

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        speaker.train(interval)   # Train interval iters

        print()
        print("Iter: %d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            bleu_score, precisions = evaluator.bleu_score(path2inst)

            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), bleu_score, idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)

            # Save the model according to the bleu score
            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                print('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

            # Screen print out
            print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))


def train(train_env, tok, n_iters, log_every=100, \
        val_envs={}, aug_env=None, augins_env=None):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)

    start_iter = 0
    #if args.load is not None:
    #    print("LOAD THE listener from %s" % args.load)
    #    start_iter = listner.load(os.path.join(args.load))

    start = time.time()

    best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}

    best_val_attacker = {'val_seen': {"accu": 1., "state":"", 'update':False},
                    'val_unseen': {"accu": 1., "state":"", 'update':False}}

    if args.fast_train:
        log_every = 40

    if args.pretrain_attacker:
        print("LOAD THE listener from %s" % args.load)
        listner.load(os.path.join(args.load))

    elif args.adv_train:
        if args.load is not None:
            print("LOAD THE listener from %s" % args.load)
            listner.load(os.path.join(args.load))
        if args.load_attacker is not None:
            print("LOAD THE attacker from %s" % args.load_attacker)
            listner.load_attacker(os.path.join(args.load_attacker))

        alter_count = 0

    elif args.finetune_agent:
        print("LOAD THE listener from %s" % args.load)
        listner.load(os.path.join(args.load))
    else:
        if args.load is not None:
           print("LOAD THE listener from %s" % args.load)
           start_iter = listner.load(os.path.join(args.load))        

    data_log = defaultdict(list)

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        data_log['iteration'].append(iter)

        # Train for log_every interval
        if aug_env is None and augins_env is None:     # The default training process
            listner.env = train_env
            if args.pretrain_agent or args.finetune_agent:
                listner.train(interval, feedback=args.feedback)
            elif args.pretrain_attacker:                    
                listner.train_attacker(interval, feedback=args.feedback, \
                    feedback_attacker=args.feedback_attacker)
            elif args.adv_train:
                # train nav at initial turn
                if idx == 0:
                    feedback_method = 'sample'
                    attacker_feedback_method = 'argmax'

                if feedback_method == 'sample' and attacker_feedback_method == 'argmax':
                    if idx % (args.iters_alter_nav + args.iters_alter_att) == args.iters_alter_nav and idx != 0:
                        alter_count += 1
                elif feedback_method == 'argmax' and attacker_feedback_method == 'sample':
                    if idx % (args.iters_alter_nav + args.iters_alter_att) == 0 and idx != 0:
                        alter_count += 1

                if alter_count % 2 == 0:
                    feedback_method = 'sample'
                    attacker_feedback_method = 'argmax'
                    print('train nav in adv train!')
                else:
                    feedback_method = 'argmax'
                    attacker_feedback_method = 'sample'
                    print('train att in adv train!')

                    listner.train_adv(interval, feedback=feedback_method, \
                        feedback_attacker=attacker_feedback_method)
            else:
                listner.train(interval, feedback=args.feedback)   # Train interval iters
        
        elif augins_env is not None:
            if args.pretrain_agent or args.finetune_agent:
                for _ in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env

                    # Train with GT data
                    args.ml_weight = 0.2
                    listner.accumulate_gradient(args.feedback)
                    listner.env = augins_env

                    # Train with Back Translation
                    args.ml_weight = 0.6  # Sem-Configuration
                    listner.accumulate_gradient(args.feedback)
                    listner.optim_step()
            elif args.pretrain_attacker:
                for _ in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env

                    # Train with GT data
                    args.ml_weight = 0.2
                    listner.accumulate_gradient( \
                        args.feedback, args.feedback_attacker)
                    listner.env = augins_env

                    # Train with Back Translation
                    args.ml_weight = 0.6  # Sem-Configuration
                    listner.accumulate_gradient( \
                        args.feedback, args.feedback_attacker)
                    listner.optim_step()
            elif args.adv_train:
                # train nav at initial turn
                if idx == 0:
                    feedback_method = 'sample'
                    attacker_feedback_method = 'argmax'

                if feedback_method == 'sample' and attacker_feedback_method == 'argmax':
                    if idx % (args.iters_alter_nav + args.iters_alter_att) == args.iters_alter_nav and idx != 0:
                        alter_count += 1
                elif feedback_method == 'argmax' and attacker_feedback_method == 'sample':
                    if idx % (args.iters_alter_nav + args.iters_alter_att) == 0 and idx != 0:
                        alter_count += 1

                if alter_count % 2 == 0:
                    feedback_method = 'sample'
                    attacker_feedback_method = 'argmax'
                    print('train nav in adv train!')

                    for _ in range(interval // 2):
                        listner.zero_grad_adv_nav()

                        listner.env = train_env

                        # Train with GT data
                        args.ml_weight = 0.2
                        listner.accumulate_gradient(feedback_method,attacker_feedback_method)

                        listner.env = augins_env

                        # Train with Back Translation
                        args.ml_weight = 0.6  # Sem-Configuration
                        listner.accumulate_gradient(feedback_method,attacker_feedback_method)
                        listner.optim_step_adv_train_nav()
                else:
                    feedback_method = 'argmax'
                    attacker_feedback_method = 'sample'
                    print('train att in adv train!')


                    for _ in range(interval // 2):
                        listner.zero_grad_adv_att()
                        listner.env = train_env
                        # Train with GT data
                        args.ml_weight = 0.2
                        listner.accumulate_gradient(feedback_method, attacker_feedback_method)

                        listner.env = augins_env

                        # Train with Back Translation
                        args.ml_weight = 0.6  # Sem-Configuration
                        listner.accumulate_gradient(feedback_method, attacker_feedback_method)
                        listner.optim_step_adv_train_att()

            else:
                assert False


        train_losses = np.array(listner.losses)

        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg
        
        if args.pretrain_agent or args.finetune_agent:

            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)
            length = max(len(listner.logs['critic_loss']), 1)
            critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
            entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
            predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
            
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/unsupervised", predict_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            print("total_actions", total)
            print("max_length", length)

            data_log['loss/critic'].append(critic_loss)
            data_log['policy_entropy'].append(entropy)
            data_log['total_actions'].append(total)
            data_log['max_length'].append(length) 

        elif args.pretrain_attacker:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total_attacker']), 1)
            length = max(len(listner.logs['critic_loss_attacker']), 1)
            critic_loss = sum(listner.logs['critic_loss_attacker']) / total #/ length / args.batchSize
            entropy = sum(listner.logs['entropy_attacker']) / total #/ length / args.batchSize
            predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
            
            writer.add_scalar("loss/critic_attacker", critic_loss, idx)
            writer.add_scalar("policy_entropy_attacker", entropy, idx)
            writer.add_scalar("loss/unsupervised", predict_loss, idx)
            writer.add_scalar("total_actions_attacker", total, idx)
            writer.add_scalar("max_length", length, idx)
            print("total_actions", total)
            print("max_length", length)            

            data_log['loss/critic'].append(critic_loss)
            data_log['policy_entropy'].append(entropy)
            data_log['total_actions'].append(total)
            data_log['max_length'].append(length) 

        elif args.adv_train:
            if feedback_method == 'sample' and attacker_feedback_method == 'argmax':
                # Log the training stats to tensorboard
                total = max(sum(listner.logs['total']), 1)
                length = max(len(listner.logs['critic_loss']), 1)
                critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
                entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
                predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
                
                writer.add_scalar("loss/critic", critic_loss, idx)
                writer.add_scalar("policy_entropy", entropy, idx)
                writer.add_scalar("loss/unsupervised", predict_loss, idx)
                writer.add_scalar("total_actions", total, idx)
                writer.add_scalar("max_length", length, idx)
                print("total_actions", total)
                print("max_length", length)

                data_log['loss/critic'].append(critic_loss)
                data_log['policy_entropy'].append(entropy)
                data_log['total_actions'].append(total)
                data_log['max_length'].append(length) 

            elif feedback_method == 'argmax' and attacker_feedback_method == 'sample':
                # Log the training stats to tensorboard
                total = max(sum(listner.logs['total_attacker']), 1)
                length = max(len(listner.logs['critic_loss_attacker']), 1)
                critic_loss = sum(listner.logs['critic_loss_attacker']) / total #/ length / args.batchSize
                entropy = sum(listner.logs['entropy_attacker']) / total #/ length / args.batchSize
                predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
                
                writer.add_scalar("loss/critic_attacker", critic_loss, idx)
                writer.add_scalar("policy_entropy_attacker", entropy, idx)
                writer.add_scalar("loss/unsupervised", predict_loss, idx)
                writer.add_scalar("total_actions_attacker", total, idx)
                writer.add_scalar("max_length", length, idx)
                print("total_actions", total)
                print("max_length", length)  

                data_log['loss/critic'].append(critic_loss)
                data_log['policy_entropy'].append(entropy)
                data_log['total_actions'].append(total)
                data_log['max_length'].append(length) 

        else:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)
            length = max(len(listner.logs['critic_loss']), 1)
            critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
            entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
            predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
            
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/unsupervised", predict_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            print("total_actions", total)
            print("max_length", length)

            data_log['loss/critic'].append(critic_loss)
            data_log['policy_entropy'].append(entropy)
            data_log['total_actions'].append(total)
            data_log['max_length'].append(length) 

        # Run validation
        #loss_str = ""
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation loss under the same conditions as training
            iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280

            # Get validation distance from goal under test evaluation conditions
            if args.pretrain_agent or args.finetune_agent:
                listner.test(use_dropout=False, feedback='argmax', iters=iters)
            elif args.pretrain_attacker:
                listner.test_attacker(use_dropout=False, feedback='argmax', \
                    feedback_attacker='argmax', iters=iters)
            elif args.adv_train:
                if feedback_method == 'sample' and attacker_feedback_method == 'argmax':
                    train_nav=True
                elif feedback_method == 'argmax' and attacker_feedback_method == 'sample':
                    train_nav=False
                listner.test_attacker(use_dropout=False, feedback='argmax', \
                    feedback_attacker='argmax', iters=iters, train_nav=train_nav)
            else:
                listner.test(use_dropout=False, feedback='argmax', iters=iters)
            
            result = listner.get_results()
            
            score_summary, _ = evaluator.score(result)
            
            loss_str += ", %s " % env_name
            for metric,val in score_summary.items():
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['success_rate']:
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if args.pretrain_agent or args.finetune_agent:
                        if env_name in best_val:                        
                            if val > best_val[env_name]['accu']:
                                best_val[env_name]['accu'] = val
                                best_val[env_name]['update'] = True
                    elif args.pretrain_attacker:        
                        if env_name in best_val_attacker: 
                            if val < best_val_attacker[env_name]['accu']:
                                best_val_attacker[env_name]['accu'] = val
                                best_val_attacker[env_name]['update'] = True
                    elif args.adv_train:
                        if feedback_method == 'sample' and attacker_feedback_method == 'argmax':
                            if env_name in best_val:                        
                                if val > best_val[env_name]['accu']:
                                    best_val[env_name]['accu'] = val
                                    best_val[env_name]['update'] = True                            
                        elif feedback_method == 'argmax' and attacker_feedback_method == 'sample':
                            if env_name in best_val_attacker: 
                                if val < best_val_attacker[env_name]['accu']:
                                    best_val_attacker[env_name]['accu'] = val
                                    best_val_attacker[env_name]['update'] = True
                    else:
                        if env_name in best_val:                        
                            if val > best_val[env_name]['accu']:
                                best_val[env_name]['accu'] = val
                                best_val[env_name]['update'] = True

                loss_str += ', %s: %.3f' % (metric, val)
                

        if args.pretrain_agent or args.finetune_agent:
            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val[env_name]['update'] = False
                    listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
        elif args.pretrain_attacker:
            for env_name in best_val_attacker:
                if best_val_attacker[env_name]['update']:
                    best_val_attacker[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val_attacker[env_name]['update'] = False
                    listner.save_attacker(idx, os.path.join("snap", args.name, "state_dict", "best_%s_attacker" % (env_name)))
        elif args.adv_train:
            if feedback_method == 'sample' and attacker_feedback_method == 'argmax':
                for env_name in best_val:
                    if best_val[env_name]['update']:
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        best_val[env_name]['update'] = False
                        listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
            elif feedback_method == 'argmax' and attacker_feedback_method == 'sample': 
                for env_name in best_val_attacker:
                    if best_val_attacker[env_name]['update']:
                        best_val_attacker[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        best_val_attacker[env_name]['update'] = False
                        listner.save_attacker(idx, os.path.join("snap", args.name, "state_dict", "best_%s_attacker" % (env_name)))
        else:
            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val[env_name]['update'] = False
                    listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))


        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        
        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            if args.pretrain_agent or args.finetune_agent:
                for env_name in best_val:
                    print(env_name, best_val[env_name]['state'])
            elif args.pretrain_attacker:
                for env_name in best_val_attacker:
                    print(env_name, best_val_attacker[env_name]['state'])                
            elif args.adv_train:
                if feedback_method == 'sample' and attacker_feedback_method == 'argmax':
                    for env_name in best_val:
                        print(env_name, best_val[env_name]['state'])                    
                elif feedback_method == 'argmax' and attacker_feedback_method == 'sample': 
                    for env_name in best_val_attacker:
                        print(env_name, best_val_attacker[env_name]['state']) 
            else:
                for env_name in best_val:
                    print(env_name, best_val[env_name]['state'])

        if iter % 50000 == 0:
            if args.pretrain_agent or args.finetune_agent:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))
            elif args.pretrain_attacker:
                listner.save_attacker(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d_attacker" % (iter)))
            elif args.adv_train:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))
                listner.save_attacker(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d_attacker" % (iter)))
            else:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

        # df = pd.DataFrame(data_log)
        # df.set_index('iteration')
        # df_path = '%s-log.csv' % (PLOT_DIR)
        # df.to_csv(df_path)
       
    if args.pretrain_agent or args.finetune_agent:
        listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))
    elif args.pretrain_attacker:
        listner.save_attacker(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d_attacker" % (idx)))
    elif args.adv_train:
        listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))
        listner.save_attacker(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d_attacker" % (idx)))
    else:
        listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))

def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def beam_valid(train_env, tok, val_envs={}):
    listener = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = Speaker(train_env, listener, tok)
    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)

    print("Loaded the listener model at iter % d" % listener.load(args.load))

    final_log = ""
    for env_name, (env, evaluator) in val_envs.items():
        listener.logs = defaultdict(list)
        listener.env = env

        listener.beam_search_test(speaker)
        results = listener.results

        def cal_score(x, alpha, avg_speaker, avg_listener):
            speaker_score = sum(x["speaker_scores"]) * alpha
            if avg_speaker:
                speaker_score /= len(x["speaker_scores"])
            # normalizer = sum(math.log(k) for k in x['listener_actions'])
            normalizer = 0.
            listener_score = (sum(x["listener_scores"]) + normalizer) * (1-alpha)
            if avg_listener:
                listener_score /= len(x["listener_scores"])
            return speaker_score + listener_score

        if args.param_search:
            # Search for the best speaker / listener ratio
            interval = 0.01
            logs = []
            for avg_speaker in [False, True]:
                for avg_listener in [False, True]:
                    for alpha in np.arange(0, 1 + interval, interval):
                        result_for_eval = []
                        for key in results:
                            result_for_eval.append({
                                "instr_id": key,
                                "trajectory": max(results[key]['paths'],
                                                  key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                                  )['trajectory']
                            })
                        score_summary, _ = evaluator.score(result_for_eval)
                        for metric,val in score_summary.items():
                            if metric in ['success_rate']:
                                print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                                      (avg_speaker, avg_listener, alpha, val))
                                logs.append((avg_speaker, avg_listener, alpha, val))
            tmp_result = "Env Name %s\n" % (env_name) + \
                    "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f\n" % max(logs, key=lambda x: x[3])
            print(tmp_result)
            # print("Env Name %s" % (env_name))
            # print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
            #       max(logs, key=lambda x: x[3]))
            final_log += tmp_result
            print()
        else:
            avg_speaker = True
            avg_listener = True
            alpha = args.alpha

            result_for_eval = []
            for key in results:
                result_for_eval.append({
                    "instr_id": key,
                    "trajectory": [(vp, 0, 0) for vp in results[key]['dijk_path']] + \
                                  max(results[key]['paths'],
                                   key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                  )['trajectory']
                })
            # result_for_eval = utils.add_exploration(result_for_eval)
            score_summary, _ = evaluator.score(result_for_eval)

            if env_name != 'test':
                loss_str = "Env Name: %s" % env_name
                for metric, val in score_summary.items():
                    if metric in ['success_rate']:
                        print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                              (avg_speaker, avg_listener, alpha, val))
                    loss_str += ",%s: %0.4f " % (metric, val)
                print(loss_str)
            print()

            if args.submit:
                json.dump(
                    result_for_eval,
                    open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    print(final_log)


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(features)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, \
        splits=['train'], tokenizer=tok, \
        load_target_and_candidate_word=args.pretrain_attacker)

    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        #val_env_names.append('train')

    if not args.beam:
        val_env_names.append("train")

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, \
                    splits=[split], tokenizer=tok, \
                    load_target_and_candidate_word=args.pretrain_attacker),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        if args.beam:
            beam_valid(train_env, tok, val_envs=val_envs)
        else:
            valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    else:
        assert False


def valid_speaker(tok, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[path_id]['instructions'])
        pathXinst = list(path2inst.items())
        name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
        score_string = " "
        for score_name, score in name2score.items():
            score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
        print("For env %s" % env_name)
        print(score_string)
        print("Average Length %0.4f" % utils.average_length(path2inst))

def train_val_augment_ins():
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(features)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    #aug_path = args.aug

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok,
                         load_target_and_candidate_word=args.pretrain_attacker)
    augins_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train', 'literal_speaker_data_augmentation_paths'], tokenizer=tok,
                             name='aug', load_target_and_candidate_word=args.pretrain_attacker)

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = augins_env.get_statistics()
    print("The augmentation data size is %d" % augins_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=None, augins_env=augins_env)

def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(features)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    aug_path = args.aug

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok, name='aug')

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    elif args.train == 'auginslistener':
        train_val_augment_ins()
    else:
        assert False

