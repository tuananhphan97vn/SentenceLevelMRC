import torch 


def pool_sequential_embed(roberta_embed , start , end , method):
    if method =='mean':
        sub_matrix = roberta_embed[start:end+1 , :] 
        return torch.mean(sub_matrix , axis = 0 ) 

def pool_particular_embed(roberta_embed , list_index , method):
    #roberta embed shape (seq_length , hidden_size)
    if method =='mean':
        result = [] 
        for index in list_index:
            result.append(roberta_embed[index])
        return torch.mean(torch.stack(result) ,  axis = 0) #shape (hidden size) 

def pool_graph_embed(roberta_embed, features , method_pool = 'mean'):
    #roberta embed of each sample in a batch: shape (sequen_length , hidden size)
    results = [] #results is list, one element in result correspond with triple value (question embed , sents_embed, unique_subword_embed) of single feature
    #features can be understood as batch of features
    batch_bound_sents = []
    for i, feature in enumerate(features):
        bound_question, bound_sents , bound_subwords = find_boudaries_single_feature(feature)
        batch_bound_sents.append(bound_sents)
        question_embed = pool_sequential_embed(roberta_embed[i] , bound_question[0] , bound_question[1] , method_pool) #shape (hidden_size)
        sents_embed = [] 
        for bound_sent in bound_sents:
            sents_embed.append(pool_sequential_embed(roberta_embed[i] , bound_sent[0] , bound_sent[1] , method_pool) )
        sents_embed = torch.stack(sents_embed , axis = 0) #shape (num_sent , hidden_size)
        unique_subword_emb = []
        for bound_subword in bound_subwords:
            unique_subword_emb.append(pool_particular_embed(roberta_embed[i] , bound_subwords[bound_subword] , method_pool))  
        unique_subword_emb = torch.stack(unique_subword_emb) #shape (num subword , hidden size)
        results.append([question_embed , sents_embed , unique_subword_emb])
    return results , batch_bound_sents

def find_boudaries_single_feature(single_feature):
    input_ids = single_feature.input_ids
    paragraph_len = single_feature.paragraph_len
    tokens = single_feature.tokens
    sentid = single_feature.sentid

    #bound question
    list_id_question = input_ids[1:len(tokens)-paragraph_len-3]
    list_id_text= input_ids[ len(tokens)-paragraph_len-1  :len(tokens)-1]
    bound_question =  [1 , len(tokens)-paragraph_len-3]

    #bound sents
    sent_tok = {}
    for i in range(len(sentid)):
        if sentid[i] not in sent_tok:
            sent_tok[sentid[i]] = [i]
        else:
            sent_tok[sentid[i]].append(i)

    bound_sents = [] 
    for sent_id in sent_tok:
        list_sub_position = sent_tok[sent_id]
        start = list_sub_position[0] + len(list_id_question) + 3
        end = list_sub_position[len(list_sub_position) - 1] + len(list_id_question) + 3
        bound_sents.append([start,end])

    #boud unique subword
    list_unique_id_text  =  [] #this list must match with order of word node in graph ws and g raph ww 
    for i in range(len(list_id_text)):
        if list_id_text[i] not in list_unique_id_text:
            list_unique_id_text.append(list_id_text[i])  
    bound_subword = {} #dictionary with key is id of subword (unique and ordered) , value is index of this subword in text (certain subword can appear more than 1 times)
    for i in range(len(tokens)-paragraph_len-1 , len(tokens)-1 , 1 ):
        if input_ids[i] not in bound_subword:
            bound_subword[input_ids[i]] = [i] 
        else:
            bound_subword[input_ids[i]].append(i) #create one list of index of this subword
    return bound_question, bound_sents , bound_subword
