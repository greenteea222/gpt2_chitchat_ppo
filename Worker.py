import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import utils

from torch.autograd import Variable
from utils import get_sentence_prob

class Worker(object):
    def __init__(self, actions_p, actions_log_p, chat_history, reward_easy_to_answer, args, device):
        self.actions_p = actions_p
        self.actions_log_p = actions_log_p
        self.chat_history = chat_history

        # self.tokenizer_path = args.tokenizer_path
        # self.mmi_model_path = args.mmi_model_path

        self.lambda_easy_answer = args.lambda_easy_answer     # 易回答性，计算agent生成语句与现有列表中的各句子相似度，并取平均，及负值，默认系数为0.25
        self.lambda_novel = args.lambda_novel           # 新颖性，计算每个agent已生成语句间的互信息，并取负值，默认系数为0.25
        self.lambda_mutual_inform = args.lambda_mutual_inform   # 计算policy(正向模型)与 mmi(负向模型) 生成当前语句的互信息（前后向概率相加），取正值 0.5

        self.args = args
        self.device = device

        self.hard_answer_seqs_list = ['不知道', '好吧', '你说什么', '什么意思', '哦', '你猜']
        self.params_size = None
        self.reward = 0.0
        self.reward_easy_to_answer = reward_easy_to_answer

    def get_reward(self, mmi_model, tokenizer, bert_model, bert_tokenizer):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        # if torch.cuda.is_available():
        #     device = torch.device(self.device)
        #     cudnn.benchmark = True
        #     cudnn.enable = True
        #     torch.cuda.manual_seed(self.args.seed)
        # else:
        #     device = torch.device('cpu')

        reward_easy_to_answer = self.lambda_easy_answer * self.reward_easy_to_answer
        reward_novel = self.lambda_novel * self.reward_novel(bert_model, bert_tokenizer)
        reward_mutal_inform = self.lambda_mutual_inform * self.reward_mutal_inform(mmi_model, tokenizer)

        # print('reward_easy_to_answer = {}, reward_novel = {}, reward_mutal_inform = {}'.format(
        #     reward_easy_to_answer, reward_novel, reward_mutal_inform
        # ))
        self.reward = reward_easy_to_answer + reward_novel + reward_mutal_inform

        return reward_easy_to_answer, reward_novel, reward_mutal_inform, self.reward

    # def reward_easy_to_answer(self):
    #     reward = 0.0
    #     for sentence in self.chat_history:
    #         for hard_sentence in self.hard_answer_seqs_list:
    #             reward += -(utils.get_sentence_prob(sentence, hard_sentence, self.model, self.tokenizer) / len(hard_sentence))
    #     reward /= (len(self.chat_history) * len(self.hard_answer_seqs_list))
    #     return reward

    def reward_novel(self, bert_model, bert_tokenizer):
        reward = torch.FloatTensor([0.0]).to(torch.device('cuda:0'))
        history_encode_list = utils.get_seq_encode_list(bert_model, bert_tokenizer, self.chat_history)
        #print('len(history_encode_list) = '.format(len(history_encode_list)))
        for i in range(len(history_encode_list) - 1):
            #print('history_encode_list[i] is {}, history_encode_list[i + 1] is {}'.format(history_encode_list[i], history_encode_list[i + 1]))
            cal_res = torch.log(torch.cos(torch.dot(history_encode_list[i], history_encode_list[i + 1]) /
                                                        (torch.norm(history_encode_list[i]) * torch.norm(history_encode_list[i + 1]))))
            #print('cal_res.shape = {}'.format(cal_res.shape))
            reward = reward - torch.log(torch.cos(torch.dot(history_encode_list[i], history_encode_list[i + 1]) /
                                                        (torch.norm(history_encode_list[i]) * torch.norm(history_encode_list[i + 1]))))
        #print('reward = {}'.format(reward))
        #print('len(history_encode_list) = {}'.format(len(history_encode_list)))
        reward = reward / (len(history_encode_list) + 1)
        return reward

    def reward_mutal_inform(self, mmi_model, tokenizer):
        reward = torch.FloatTensor([0.0]).to(torch.device('cuda:0'))
        for i in range(1, len(self.chat_history) - 1):
            #p_pos = self.actions_p[i]
            sentence1 = self.chat_history[i]
            sentence2 = ''.join(self.chat_history[i:])
            _, log_p_reverse = utils.get_sentence_prob(sentence1, sentence2, mmi_model, tokenizer, is_reverse = True)
            reward = reward + (self.actions_log_p[i] + log_p_reverse) / (len(sentence1) + 1)
        reward = reward / (len(self.chat_history) + 1)
        # reward = reward.cpu().float()
        return reward


