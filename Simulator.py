from torch.autograd import Variable
from transformers import BertTokenizer, GPT2LMHeadModel, BertModel
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import utils
import torch.nn.functional as F
import os
import torch.backends.cudnn as cudnn

#model = GPT2LMHeadModel.from_pretrained('dialogue_model_train/from_initial_GPT2/model_epoch10/')
#tokenizer = BertTokenizer(vocab_file = 'vocabulary/vocab_small.txt')

class Simulator(nn.Module):
    def __init__(self, args, device):
        super(Simulator, self).__init__()
        self.device = device
        self.actions_p = 1.0
        self.actions_log_p = 0.0
        self.model = GPT2LMHeadModel.from_pretrained(args.origin_model_path).to(device)
        self.tokenizer = BertTokenizer(vocab_file = args.tokenizer_path)
        self.model.resize_token_embeddings(len(self.tokenizer))


        self.hard_answer_seqs_list = ['不知道', '好吧', '你说什么', '什么意思', '哦', '你猜']

    def forward(self, input):
        output = self.model.forward(input)
        return output

    def sample(self, max_len, chats_rounds, save_samples_path=None):
        actions_p, actions_log_p, chat_history = [], [], []
        reward_easy_to_answer = 0.0
        if save_samples_path:
            if not os.path.exists(save_samples_path):
                os.makedirs(save_samples_path)
            samples_file = open(save_samples_path + '/samples.txt', 'a', encoding='utf8')
            samples_file.write("采样记录{}:\n".format(datetime.now()))

        if save_samples_path:
            samples_file.write("Begin: \n")

        input_ids = [self.tokenizer.cls_token_id]  # 每个input以[CLS]为开头

        chat_index_history = []
        # chat_index_history.append(self.tokenizer.encode('谢谢你所做的一切'))

        agent_name = ''
        for cnt in range(2 * chats_rounds):

            if (len(chat_index_history) != 0):
                input_ids.extend(chat_index_history[-1])
                input_ids.append(self.tokenizer.sep_token_id)

            # input_ids长度最长为300
            input_ids = input_ids[-min(299, len(input_ids)):]
            curr_input_tensor = torch.LongTensor(input_ids).to(self.device)
            # print('input_sentence is {}'.format(self.tokenizer.convert_ids_to_tokens(input_ids)))

            generated = []

            prob = torch.FloatTensor([1.0]).to(torch.device('cuda:0'))
            log_prob = 1.0
            # 最多生成max_len个token
            for index in range(max_len):
                # print('curr_input_tensor.size() = {}'.format(curr_input_tensor.size()))
                outputs = self.forward(input=curr_input_tensor)
                # outputs[0]: len * num_vocab
                # next_token_logits = outputs[0][-1, :].detach()
                next_token_logits = outputs[0][-1, :].clone()

                topk = 20
                topp = 0.95

                # # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                # repetition_penalty = 2
                # for id in set(generated):
                #     next_token_logits[id] /= repetition_penalty

                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = utils.top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)

                # print('filtered_logits = {}'.format(filtered_logits))


                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token_index = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                # # 贪心计算模型输出的下一个字(词)
                # next_token_index = torch.argmax(next_token_logits)

                # 根据条件概率计算 agent(i) 生成完整语句的条件概率
                #next_token_p_logits = F.softmax(next_token_logits).detach().cpu().numpy()
                next_token_p_logits = F.softmax(next_token_logits)
                prob = prob * next_token_p_logits[next_token_index]

                # 遇到[SEP]则表明response生成结束
                if next_token_index == self.tokenizer.sep_token_id:
                    if index == 0:
                        index -= 1
                        continue
                    else:
                        break
                generated.append(next_token_index.item())

                # # 贪心策略
                # curr_input_tensor = torch.cat((curr_input_tensor, next_token_index.unsqueeze(0)), dim=0)

                # topk, topp策略
                curr_input_tensor = torch.cat((curr_input_tensor, next_token_index), dim=0)

            actions_p.append(prob)
            actions_log_p.append(torch.log(prob))
            chat_index_history.append(generated)
            sentence = ('').join(self.tokenizer.convert_ids_to_tokens(generated))
            chat_history.append(sentence)

            #在这里计算 每个Worker类的易回答性奖励函数，避免在其他地方重复调用模型

            reward_easy_to_answer_tmp = 0.0
            for hard_sentence in self.hard_answer_seqs_list:
                _, log_p_2 = utils.get_sentence_prob(sentence, hard_sentence, self.model, self.tokenizer)
                reward_easy_to_answer_tmp = reward_easy_to_answer_tmp - (log_p_2 / (len(hard_sentence) + 1))

            # 只有当生成句子长度大于0(不是只生成一个[SEP])时，才计算
            if len(sentence) > 0:
                reward_easy_to_answer = reward_easy_to_answer + reward_easy_to_answer_tmp / len(sentence)

            # print(generated)
            text = self.tokenizer.convert_ids_to_tokens(generated)
            if cnt & 1 == 0:
                agent_name = 'agent1: '
            else:
                agent_name = 'agent2: '
            if save_samples_path:
                samples_file.write(agent_name + "{}\n".format("".join(text)))
        reward_easy_to_answer = reward_easy_to_answer / (2 * chats_rounds)

        actions_p = torch.Tensor(actions_p).cuda()
        actions_log_p = torch.Tensor(actions_log_p).cuda()

        #print('chat_history is {}'.format(chat_history))

        return actions_p, actions_log_p, chat_history, reward_easy_to_answer

    def get_p(self, chat_history, chats_rounds):
        assert len(chat_history) == 2 * chats_rounds

        chat_history[0] = self.tokenizer.cls_token + chat_history[0]

        actions_p, actions_log_p = [], []
        prob = torch.FloatTensor([1.0]).to(torch.device('cuda:0'))
        log_prob = 1.0
        for cnt in range(2 * chats_rounds):
            if cnt == 0:
                sentence = chat_history[0]
                for i in range(1, len(sentence)):
                    input_ids = self.tokenizer.encode(sentence[:i])
                    curr_input_tensor = torch.LongTensor(input_ids).cuda()
                    next_tokens_outputs = self.forward(input=curr_input_tensor)  # next_tokens_outputs[0]: len * num_vocab
                    #next_token_logits = next_tokens_outputs[0][-1, :].detach()
                    next_token_logits = next_tokens_outputs[0][-1, :].clone()

                    #next_token_p_logits = F.softmax(next_token_logits).detach().cpu().numpy()
                    next_token_p_logits = F.softmax(next_token_logits)
                    prob = prob * next_token_p_logits[self.tokenizer.convert_tokens_to_ids(sentence[i])]
                log_prob = torch.log(prob)
            else:
                prob, log_prob = utils.get_sentence_prob(chat_history[cnt - 1], chat_history[cnt],
                                                         self.model, self.tokenizer)

            actions_p.append(prob)
            actions_log_p.append(log_prob)
        actions_p = torch.Tensor(actions_p).cuda()
        actions_log_p = torch.Tensor(actions_log_p).cuda()
        return actions_p, actions_log_p

