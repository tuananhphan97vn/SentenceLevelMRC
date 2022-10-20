

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class Classifier(nn.Module):

	def __init__(self, sent_dim, word_dim ,  device):
		super().__init__()
		self.device = device
		self.sent_dim = sent_dim
		self.word_dim = word_dim
		self.num_layer_lstm = 2
		self.S2Wlayer = nn.Linear(self.sent_dim , self.word_dim , bias = True) #transform sentence object to word object 
		self.W2Wlayer = nn.Linear(self.word_dim , self.word_dim, bias = True) #transform word to word object
		self.W1 = nn.Linear(2* self.word_dim , self.word_dim , bias = True)
		self.W2 = nn.Linear(2 * self.word_dim ,  self.word_dim , bias = True )
		self.S_layer = nn.LSTM(input_size = self.word_dim, hidden_size=self.word_dim, num_layers=self.num_layer_lstm, batch_first=True,bidirectional= True)
		self.E_layer = nn.LSTM(input_size = self.word_dim, hidden_size=self.word_dim, num_layers=self.num_layer_lstm, batch_first=True,bidirectional= True)

	def forward(self, ques_emb , sents_emb , words_emb, length_passage):
		#ques emb shape (bs , sent dim ). 
		#sents emb is list, len(list) = bs, each element in list can have different shape depend on length of passage 
		#each element is tensor has shape (N-sents , sent dim )
		#words_emb: words_emb is in form of list, len(list) = bs 
		#each element in list is list, len(list) = number sent in one passage 
		#each element in sublist is tensor, this tensor has shape (N_words in this sent, word dim )
		#note: word dim can be different with sent dim. In that case, we propose sent dim = 64 while this value for words is 768 (setting of pretrained model) [ [(N_word , word_dim)] ]
		#to use effectively LSTM model, we must convert words_emb to tensor form. Therefore, we need to pad process due to each sentence can have different length.
		max_num_words = 0 
		for p in words_emb:
			for s in p :
				if max_num_words < s.shape[0]:
					max_num_words = s.shape[0]
		#padd sent with max num words 
		pad_word_embs = [] 
		for i in range(len(words_emb)):
			pad_passage = []
			for j in range(len(words_emb[i])):
				sent_j = words_emb[i][j] #sent j shape (N_words , word dim)
				pad_sent_j = torch.cat( ( sent_j, torch.zeros( max_num_words -  sent_j.shape[0] , self.word_dim).to(torch.device('cuda'))) , dim = 0)  #shape (max words , word dim )
				pad_passage.append(pad_sent_j) 
			pad_passage = torch.stack(pad_passage) #shape (N_sents , max words , word dim )
			pad_word_embs.append(torch.tanh(self.W2Wlayer(pad_passage)))
		#pad_word_embs is list , each element in list is tensor, which has shape as (N_sents, max words , word dim ) 

		#create input for LSTM model 
		#x_i = tanh(Linear(word_i||curr_sent)) 
		#current memory = tanh(linear(question))
		#push current memory versus x_i to LSTM model, get final hidden state 
		#input 
		sents_embed_trans = [torch.tanh(self.S2Wlayer(t)).unsqueeze(1) for t in sents_emb] #list of elements, each has shape (N_sents, 1 , word dim )
		sents_embed_trans = [ t.repeat(1, max_num_words , 1) for t in sents_embed_trans] #each one: (N_sents, max num words, words dim)
		word_sents_embed_trans = []
		for i in range(len(sents_embed_trans)):
			#concate and feed them to linear layer and non-linear activation 
			word_sent_cat = torch.cat(( pad_word_embs[i] , sents_embed_trans[i])  , dim = 2) #shape (N-sent, max word , 2* word dim )
			word_sent_cat = torch.tanh(self.W1(word_sent_cat)) #shape (N_sents , max_word , word dim )
			word_sents_embed_trans.append(word_sent_cat)
		inputs_lstm = torch.cat(word_sents_embed_trans , dim = 0 ) #shape (bs * N_sents, max word , word dim )

		#cell memory: question information 
		ques_emb_trans = torch.tanh(self.S2Wlayer(ques_emb)) #shape (bs , word dim )
		ques_cell_memory = [] 
		for i in range(ques_emb_trans.shape[0]):
			N_sents = length_passage[i] #passage length indicate number of sentence in this passage 
			duplicate_ques_emb = ques_emb_trans[i].unsqueeze(0).repeat( N_sents , 1) #shape (N sent , word dim )
			ques_cell_memory.append(duplicate_ques_emb)
		cell_lstm = torch.cat(ques_cell_memory) #shape (bs * N_sents , word dim ) 

		cell_lstm = torch.stack( [cell_lstm]* 2 * self.num_layer_lstm) # shape (4 , bs x N_sents , word dim ) 
		h_lstm = cell_lstm 

		#feed inputs_lstm and initial cell lstm through 2 lstm model: span prediction 
		output_start, (_, _) = self.S_layer(inputs_lstm, (h_lstm, cell_lstm)) #output start and ouput end shape (bs x N_sents , max words , 2 * word dim ) in this case, we set hidden dim = word dim 
		output_end , (_,_) =self.E_layer(inputs_lstm , (h_lstm , cell_lstm))

		#after pass through 2 lstm models, we need recover fform of output to match it with odinary form 
		#In other words, we must remove all of pad word token and concate them to have original orders
		output_start = torch.tanh(self.W2(output_start)) #shape( bs x N_sents , max word , word dim)
		output_end = torch.tanh(self.W2(output_end)) #shape (bs x N_sents , max word , word dim )

		original_form_start, original_form_end = [] , [] 
		split_out_start = torch.split( output_start , length_passage)
		split_out_end = torch.split(output_end , length_passage) 
		#after splitting, original out start/end is list, each element in list has shape (N_sents, max word , word dim)
		for i in range(len(split_out_start)): #for each passage 
			out_sample_start , out_sample_end = [] , [] 
			for j in range(len(split_out_start[i])):
				len_sent = words_emb[i][j].shape[0]
				out_sample_start.append(split_out_start[i][j][:len_sent, :]) #split_out_start[i][j][:len_sent] (N_words, word dim )
				out_sample_end.append(split_out_end[i][j][:len_sent , :]) 
			out_sample_start = torch.cat(out_sample_start , dim = 0 ) #shape (N_word in paragraph , word dim)
			out_sample_end  = torch.cat(out_sample_end , dim = 0 ) #shape (N word in paragraph , word dim )
			original_form_start.append(out_sample_start)
			original_form_end.append(out_sample_end)
			
		return original_form_start , original_form_end 
		

		
