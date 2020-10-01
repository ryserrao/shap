from transformers.file_utils import ModelOutput
import torch
import numpy as np

def get_encoder_outputs(input_ids,model,attention_mask):
    encoder = model.get_encoder()
    encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)
    return encoder_outputs

def cal_conditional_logits(input_ids,model,tokenizer,decoder_inputs,encoder_outputs=None,attention_mask=None):
    conditional_logits=[]
    past=None
    if attention_mask is None:
        attention_mask = (input_ids!=tokenizer.pad_token_id).type(torch.int64).cuda()
    if encoder_outputs is None:
        encoder_outputs = get_encoder_outputs(input_ids,model,attention_mask)
    for i in range(1,decoder_inputs.shape[1]):
        model_inputs = model.prepare_inputs_for_generation(
                    decoder_inputs[:,:i], past=past, attention_mask=attention_mask, use_cache=True, encoder_outputs=encoder_outputs
                )
        outputs = model(**model_inputs, return_dict=True)
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        probs=next_token_logits[0].softmax(dim=0)
        probs=probs.detach().cpu().numpy()
        #val = sp.special.logit(probs)
        conditional_logits.append(probs[decoder_inputs[0,i].item()])
    return np.array(conditional_logits)