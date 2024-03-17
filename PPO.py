from Simulator import Simulator
from Worker import Worker
import numpy as np
import torch
import torch.optim as optim
import logging
from multiprocessing import Process, Queue
import multiprocessing
import transformers
from transformers import BertTokenizer, GPT2LMHeadModel, BertModel
from datetime import datetime
import os

multiprocessing.set_start_method('spawn', force=True)


class PPO(object):
    def __init__(self, args, device, logger, exp_dir):
        self.args = args
        self.logger = logger
        self.exp_dir = exp_dir
        self.origin_model_path = args.origin_model_path

        #model = GPT2LMHeadModel.from_pretrained('dialogue_model_train/from_initial_GPT2/model_epoch10/')
        self.mmi_model = GPT2LMHeadModel.from_pretrained(args.mmi_model_path).cuda()
        self.tokenizer = BertTokenizer(vocab_file=args.tokenizer_path)
        self.bert_model = BertModel.from_pretrained(args.bert_model_path).cuda()
        self.bert_tokenizer = BertTokenizer(vocab_file=args.bert_tokenizer_path)

        self.device = device
        self.epochs = args.rl_epochs
        self.lr = args.rl_lr
        self.episodes = args.episodes
        self.entropy_weight = args.entropy_weight

        self.ppo_epochs = args.ppo_epochs

        self.simulator = Simulator(args ,device=device).to(device)

        self.adam = optim.Adam(params=self.simulator.parameters(), lr=self.lr)

        self.baseline = None
        self.baseline_weight = self.args.baseline_weight

        self.clip_epsilon = 0.2
        self.chats_rounds = 2
        self.max_len = 20

    def solve_environment(self):
        model_save_dir = os.path.join(self.exp_dir, 'model')
        sample_path = os.path.join(self.exp_dir, 'sample')

        optimizer = transformers.AdamW(self.simulator.model.parameters(), lr=self.lr, correct_bias=True)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        # 记录 out of memory的次数
        oom_time = 0

        for epoch in range(self.epochs):

            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            try:
                epoch_start_time = datetime.now()
                workers = []
                for episode in range(self.episodes):
                    # actions_p : (self.chats_rounds, ) , 其中[0,2,4,6,8, ...]为模型生成agent1的5句话的前向概率,
                    # [1,3,5,7,9, ...]为模型生成agent2的前chats_rounds句话的前向概率
                    # chat_history : 存储每个episode生成的语句词向量, (self.chats_rounds * 2, max_len), 其中max_len为每句话的长度
                    actions_p, actions_log_p, chat_history, reward_easy_to_answer = self.simulator.sample(self.max_len, self.chats_rounds, sample_path)

                    workers.append(Worker(actions_p, actions_log_p, chat_history, reward_easy_to_answer, self.args, self.device))

                #
                for ppo_epoch in range(self.ppo_epochs):
                    loss = torch.FloatTensor([0.0]).to(torch.device('cuda:0')).requires_grad_()
                    reward1, reward2, reward3 = 0.0, 0.0, 0.0
                    for worker in workers:
                        r1, r2, r3, self.baseline = worker.get_reward(self.mmi_model, self.tokenizer, self.bert_model,
                                                                      self.bert_tokenizer)
                        if ppo_epoch == 0:
                            self.baseline = 0
                        actions_p, actions_log_p = self.simulator.get_p(worker.chat_history, self.chats_rounds)
                        loss_now = self.cal_loss(actions_p, actions_log_p, worker, self.baseline)
                        loss = loss + loss_now

                        reward1 = reward1 + r1 / len(workers)
                        reward2 = reward2 + r2 / len(workers)
                        reward3 = reward3 + r3 / len(workers)
                        print('epoch {:0>3d} ppo_epoch {:0>3d} loss {:.4f}, '
                              'reward_easy_to_answer = {}, '
                              'reward_novel {}, reward_mutal_inform {} '.format(epoch, ppo_epoch, loss.item(), reward1.item(), reward2.item(), reward3.item()))
                    loss = loss / len(workers)

                    #print('epoch {:0>3d} ppo_epoch {:0>3d} loss {:.4f}'.format(epoch, ppo_epoch, loss.item()))
                    self.logger.info('epoch {:0>3d} ppo_epoch {:0>3d} loss {:.4f}'.format(epoch, ppo_epoch, loss.item()))

                    loss.backward(retain_graph=True)
                    # 更新参数
                    optimizer.step()
                    # 清空梯度信息
                    optimizer.zero_grad()

                # loss.backward()
                # # 清空梯度信息
                # optimizer.zero_grad()

                # for worker in workers:
                #     r1, r2, r3, self.baseline = worker.get_reward(self.mmi_model, self.tokenizer, self.bert_model,
                #                                                   self.bert_tokenizer)
                #
                #     reward1 = reward1 + r1 / len(workers)
                #     reward2 = reward2 + r2 / len(workers)
                #     reward3 = reward3 + r3 / len(workers)
                # print('epoch {:0>3d} ppo_epoch {:0>3d} loss {:.4f}, reward_easy_to_answer = {}, '
                #              'reward_novel {}, reward_mutal_inform {} '.format(epoch, ppo_epoch, loss,
                #                                                                        reward1, reward2, reward3))
                # self.logger.info('epoch {:0>3d} ppo_epoch {:0>3d} loss {:.4f}, reward_easy_to_answer = {}, '
                #              'reward_novel {}, reward_mutal_inform {} '.format(epoch, ppo_epoch, loss,
                #                                                                        reward1, reward2, reward3))

                # #保存模型
                # self.logger.info('saving model for epoch {}'.format(epoch + 1))
                # model_path = os.path.join(model_save_dir, 'model_epoch{}'.format(epoch + 1))
                # if not os.path.exists(model_path):
                #     os.mkdir(model_path)
                # model_to_save = self.simulator.model.module if hasattr(self.simulator.model, 'module') else self.simulator.model
                # model_to_save.save_pretrained(model_path)

                self.logger.info('epoch {} finished'.format(epoch + 1))
                epoch_finish_time = datetime.now()
                self.logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    self.logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    self.logger.info(str(exception))
                    raise exception


    def clip(self, actions_importance):
        lower = torch.ones_like(actions_importance).to(self.device) * (1 - self.clip_epsilon)
        upper = torch.ones_like(actions_importance).to(self.device) * (1 + self.clip_epsilon)

        actions_importance, _ = torch.min(torch.cat([actions_importance.unsqueeze(0), upper.unsqueeze(0)], dim=0), dim=0)
        actions_importance, _ = torch.max(torch.cat([actions_importance.unsqueeze(0), lower.unsqueeze(0)], dim=0), dim=0)

        return actions_importance

    def cal_loss(self, actions_p, actions_log_p, worker, baseline):
        actions_importance = actions_p / worker.actions_p
        clipped_actions_importance = self.clip(actions_importance)

        reward = worker.reward
        #reward = worker.reward - baseline

        actions_reward = actions_importance * reward
        clipped_actions_reward = clipped_actions_importance * reward

        actions_reward, _ = torch.min(torch.cat([actions_reward.unsqueeze(0), clipped_actions_reward.unsqueeze(0)], dim=0), dim=0)
        policy_loss = -1 * torch.sum(actions_reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight

        # print('policy_loss = {}, entropy_bonus = {}'.format(policy_loss, entropy_bonus))
        #return policy_loss + entropy_bonus

        return policy_loss

