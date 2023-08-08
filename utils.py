import torch

def tokenize_and_mask_for_translation(trg, trg_tokenizer, max_length, device):
    def attention_mask(size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
        return mask == 0
    
    trg_pad_token = torch.tensor(trg_tokenizer.convert_tokens_to_ids(["[PAD]"]), dtype=torch.int64)
    trg_num_padding_tokens = max_length - len(trg)
    
    dec_input = torch.cat(
            [
                trg,  
                torch.tensor([trg_pad_token] * trg_num_padding_tokens, dtype=torch.int64)
            ]
        ).unsqueeze(0)
    
    mask = (dec_input != trg_pad_token).unsqueeze(0).unsqueeze(0).int() & attention_mask(dec_input.size(1)) 
    
    return {
        "dec_input": dec_input,
        "mask": mask
    }
