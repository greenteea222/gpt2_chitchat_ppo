import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn import manifold, datasets
from datetime import datetime
from transformers import BertTokenizer, GPT2LMHeadModel, BertModel
from time import time

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def normalize(v):
    #norm = np.linalg.norm(v)
    norm = torch.norm(v)
    if norm == 0:
       return v
    return v / norm

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # values, indices = torch.topk(logits, top_k)
        # torch.topk(logits, top_k)[0][..., -1, None] = values[..., -1, None], 表示topK中最小的元素, 用作筛选
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def get_sentence_prob(sentence1, sentence2, model, tokenizer, is_reverse = False):
    '''
    获得句子的条件概率
    :param his_sentence: 包含当前新句子的所有对话记录
    :param now_index: 当前句子的起始下标
    :param model: GPT2LMHeadModel语言模型
    :param tokenizer: BertTokenizer类型的词表类
    :param is_reverse: 输入句子是否需要为反向,若反向，则为 reverse(sentence1 + sentence2)
    :return:
    '''
    prob = torch.FloatTensor([1.0]).to(torch.device('cuda:0'))
    log_prob = 1.0

    if len(sentence1) == 0 or len(sentence2) == 0:
        return 0.0, 0.0

    sentence = sentence1 + sentence2
    start_index = len(sentence1)
    if is_reverse:
        sentence = sentence[::-1]
        start_index = len(sentence2)
    #print(sentence)
    # print('vocab size = {}'.format(tokenizer.vocab_size))
    for i in range(start_index, len(sentence)):
        input_ids = tokenizer.encode(sentence[:i])
        curr_input_tensor = torch.LongTensor(input_ids).to(torch.device('cuda:0'))
        next_tokens_outputs = model(input_ids=curr_input_tensor)   # next_tokens_outputs[0]: len * num_vocab
        # next_token_logits = next_tokens_outputs[0][-1, :].detach()
        # next_token_p_logits = F.softmax(next_token_logits).cpu().numpy()

        next_token_logits = next_tokens_outputs[0][-1, :]
        next_token_p_logits = F.softmax(next_token_logits)

        prob = prob * next_token_p_logits[tokenizer.convert_tokens_to_ids(sentence[i])]

        # next_token_log_p_logits = F.log_softmax(next_token_logits).numpy()
        # log_prob *= next_token_log_p_logits[tokenizer.convert_tokens_to_ids(sentence[i])]
    # return prob, log_prob
    # if not torch.gt(prob, 1e-4):
    #     prob = torch.FloatTensor([1e-4]).to(torch.device('cuda:0'))
    return prob, torch.log(prob + 1e-9)

def get_and_print_sim_matrix(data):
    sim_matrix = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            sim_matrix[i][j] = np.mean(np.dot(data[i], data[j]))
    print(sim_matrix)

def tsne_and_show(data, n_components = 2):
    tsne = TSNE(n_components)
    Y = tsne.fit_transform(data)
    Y[:, 0] = normalize(Y[:, 0])
    Y[:, 1] = normalize(Y[:, 1])
    # print(tsne.embedding_)
    plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
    for i in range(len(Y[:, 0])):
        plt.annotate(hard_answer_seqs_list[i], xy = (Y[:, 0][i], Y[:, 1][i]), xytext = (Y[:, 0][i]+0.01, Y[:, 1][i]+0.01)) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
    plt.show()

def get_seq_encode_list(bert_model, bert_tokenizer, seqs_list):
    #tmp = torch.LongTensor([]).cuda()
    tmp = []
    for i in range(len(seqs_list)):
        curr_input_tensor = torch.LongTensor(bert_tokenizer.encode(seqs_list[i])).unsqueeze(0).cuda()
        #print('curr_input_tensor.shape = {}'.format(curr_input_tensor.shape))
        if curr_input_tensor.shape[1] > 0:
            outputs = bert_model(curr_input_tensor)

            seq_vec = outputs[1].squeeze(0)
            # tmp = torch.cat((tmp.clone(), normalize(seq_vec)), -1)

            #norm_tmp = normalize(seq_vec)
            #print('norm_tmp.shape = {}'.format(norm_tmp.shape))
            tmp.append(normalize(seq_vec))

            # print(curr_input_tensor.numpy())

    #print('tmp.shape = {}'.format(tmp.shape))
    #tmp = torch.cat(tuple(tmp), 0)
    # encode_list = tmp.squeeze(1)
    # print('len(tmp) = {}'.format(len(tmp)))
    return tmp

def sample(model, tokenizer, max_len, chats_rounds, device, save_samples_path = None):
    model = model.to(device)
    actions_p, actions_log_p, chat_history = [], [], []
    if save_samples_path:
        if not os.path.exists(save_samples_path):
            os.makedirs(save_samples_path)
        samples_file = open(save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("采样记录{}:\n".format(datetime.now()))

    if save_samples_path:
        samples_file.write("Begin: \n")

    input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

    chat_index_history = []
    chat_index_history.append(tokenizer.encode('谢谢你所做的一切'))

    agent_name = ''
    for cnt in range(2 * chats_rounds):

        if (len(chat_index_history) != 0):
            input_ids.extend(chat_index_history[-1])
            input_ids.append(tokenizer.sep_token_id)

        #input_ids长度最长为300
        input_ids = input_ids[-min(299, len(input_ids)):]
        curr_input_tensor = torch.LongTensor(input_ids).to(device)
        #print('input_sentence is {}'.format(tokenizer.convert_ids_to_tokens(input_ids)))

        generated = []

        prob = 1.0
        log_prob = 1.0
        # 最多生成max_len个token
        for index in range(max_len):
            #print('curr_input_tensor.size() = {}'.format(curr_input_tensor.size()))
            outputs = model.forward(input_ids=curr_input_tensor)
            # outputs[0]: len * num_vocab
            next_token_logits = outputs[0][-1, :].detach()

            '''
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            repetition_penalty = 2
            topk = 20
            topp = 0.98
            for id in set(generated):
                next_token_logits[id] /= repetition_penalty

            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token_index = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            '''
            # # 贪心计算模型输出的下一个字(词)
            next_token_index = torch.argmax(next_token_logits)

            # 根据条件概率计算 agent(i) 生成完整语句的条件概率
            next_token_p_logits = F.softmax(next_token_logits).cpu().numpy()
            prob = prob *next_token_p_logits[next_token_index]
            # next_token_log_p_logits = F.log_softmax(next_token_logits).cpu().numpy()
            # log_prob *= next_token_log_p_logits[next_token_index]
            # log_prob *= np.log(next_token_p_logits[next_token_index])

            # 遇到[SEP]则表明response生成结束
            if next_token_index == tokenizer.sep_token_id:
                break
            generated.append(next_token_index.item())

            #贪心策略
            curr_input_tensor = torch.cat((curr_input_tensor, next_token_index.unsqueeze(0)), dim = 0)

            #topk, topp策略
            #curr_input_tensor = torch.cat((curr_input_tensor, next_token_index), dim=0)


        actions_p.append(prob)
        actions_log_p.append(np.log(prob))
        chat_index_history.append(generated)
        chat_history.append(('').join(tokenizer.convert_ids_to_tokens(generated)))

        # print(generated)
        text = tokenizer.convert_ids_to_tokens(generated)
        if cnt & 1 == 0:
            agent_name = 'agent1: '
        else:
            agent_name = 'agent2: '
        if save_samples_path:
            samples_file.write(agent_name + "{}\n".format("".join(text)))

        print(agent_name + "".join(text))

    return actions_p, actions_log_p, chat_history

hard_answer_seqs_list = ['不知道', '好吧', '你说什么', '什么意思', '哦', '你猜']

if __name__ == "__main__":

    '''
    model = GPT2LMHeadModel.from_pretrained('dialogue_model_train/from_initial_GPT2/model_epoch10/')
    tokenizer = BertTokenizer(vocab_file='vocabulary/vocab_small.txt')
    model.eval()

    bert_model = BertModel.from_pretrained('BERT_chinese/model')
    bert_tokenizer = BertTokenizer(vocab_file='BERT_chinese/model/vocab.txt')
    bert_model.eval()
    '''
    pass




