import torch
import numpy as np
import scipy as sp
from transformers import AutoTokenizer, AutoModelWithLMHead

class GenerateLogits:
    def __init__(self,model="distilgpt2",tokenizer=None):
        if isinstance(model,str):
            # load model and tokenizer from transformers library
            self.model = AutoModelWithLMHead.from_pretrained(model)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.model = model
            self.tokenizer = tokenizer
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        null_tokens = self.tokenizer.encode("")

        if len(null_tokens) == 0:
            self.keep_prefix = 0
            self.keep_suffix = 0
        elif len(null_tokens) == 1:
            null_token = null_tokens[0]
            assert (('eos_token' in self.tokenizer.special_tokens_map) or ('bos_token' in self.tokenizer.special_tokens_map)), "No eos token or bos token found in tokenizer!"
            if ('eos_token' in self.tokenizer.special_tokens_map) and (self.tokenizer.decode(null_token) == self.tokenizer.special_tokens_map['eos_token']):
                self.keep_prefix = 0
                self.keep_suffix = 1
            elif ('bos_token' in self.tokenizer.special_tokens_map) and (self.tokenizer.decode(null_token) == self.tokenizer.special_tokens_map['bos_token']):
                self.keep_prefix = 1
                self.keep_suffix = 0
        else:
            assert len(null_tokens) % 2 == 0, "An odd number of boundary tokens are added to the null string!"
            self.keep_prefix = len(null_tokens) // 2
            self.keep_suffix = len(null_tokens) // 2

    def get_teacher_forced_logits(self,source_sentence_ids,target_sentence_ids):
        self.model.eval()
        if self.model.config.is_encoder_decoder:
            # assigning decoder start token id 
            decoder_start_token_id = None

            if hasattr(self.model.config, "decoder_start_token_id") and self.model.config.decoder_start_token_id is not None:
                decoder_start_token_id = self.model.config.decoder_start_token_id
            elif hasattr(self.model.config, "bos_token_id") and self.model.config.bos_token_id is not None:
                decoder_start_token_id = self.model.config.bos_token_id
            elif (hasattr(self.model.config, "decoder") and hasattr(self.model.config.decoder, "bos_token_id") and self.model.config.decoder.bos_token_id is not None):
                decoder_start_token_id = self.model.config.decoder.bos_token_id
            else:
                raise ValueError(
                    "No decoder_start_token_id or bos_token_id defined in config for encoder-decoder generation"
                )

            target_sentence_start_id = torch.tensor([[decoder_start_token_id]]).to(self.device)
            target_sentence_ids = torch.cat((target_sentence_start_id,target_sentence_ids),dim=-1)
            with torch.no_grad():
                outputs = self.model(input_ids=source_sentence_ids, decoder_input_ids=target_sentence_ids, labels=target_sentence_ids, return_dict=True)
            logits=outputs.logits.detach().cpu().numpy()
        else:
            combined_sentence_ids = torch.cat((source_sentence_ids,target_sentence_ids),dim=-1)
            with torch.no_grad():
                outputs = self.model(input_ids=combined_sentence_ids, return_dict=True)
            logits=outputs.logits.detach().cpu().numpy()[:,source_sentence_ids.shape[1]-1:,:]
        del outputs
        return logits

    def get_sentence_ids(self, source_sentence,target_sentence):
        source_sentence_ids = torch.tensor([self.tokenizer.encode(source_sentence)])
        if self.keep_suffix > 0:
            target_sentence_ids = torch.tensor([self.tokenizer.encode(target_sentence)])[:,self.keep_prefix:-self.keep_suffix]
        else:
            target_sentence_ids = torch.tensor([self.tokenizer.encode(target_sentence)])[:,self.keep_prefix:]
        return source_sentence_ids.to(self.device), target_sentence_ids.to(self.device)

    def get_output_names(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def generate(self,source_sentence, target_sentence):
        source_sentence_ids, target_sentence_ids = self.get_sentence_ids(source_sentence,target_sentence)
        logits = self.get_teacher_forced_logits(source_sentence_ids,target_sentence_ids)
        conditional_logits = []
        for i in range(0,logits.shape[1]-1):
            probs = (np.exp(logits[0][i]).T / np.exp(logits[0][i]).sum(-1)).T
            logit_dist = sp.special.logit(probs)
            conditional_logits.append(logit_dist[target_sentence_ids[0,i].item()])
        return np.array(conditional_logits)