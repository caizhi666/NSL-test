import numpy as np
import torch
import time
import math

torch.set_printoptions(8)

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / math.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    pass


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)  
    x_normalized = (x - mean) / torch.sqrt(var + eps)
    return g * x_normalized + b
    pass

def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']
    w=torch.Tensor(w)
    b=torch.Tensor(b)
    return torch.matmul(x,w)+b
    pass
    

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']

    # b1,w1=w_b1['b'],w_b1['w']
    # b2,w2=w_b2['b'],w_b2['w']
    # b1=torch.Tensor(b1)
    # b2=torch.Tensor(b2)
    # w1=torch.Tensor(w1)
    # w2=torch.Tensor(w2)
    return linear(gelu(linear(x,w_b1)),w_b2)
    pass


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    d_k=k.size(-1)
    score=torch.matmul(q,torch.transpose(k,0,1))/torch.sqrt(torch.tensor(d_k,dtype=q.dtype,device=q.device))
    score=score.masked_fill(mask==-1e9,-1e9)
    weight=softmax(score)
    return torch.matmul(weight,v)
    pass

def mha(x, attn, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """

    qkv = torch.chunk(x,3,dim=-1) # need to modify

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    n_seq=x.size(dim=0)
    causal_mask = torch.triu(torch.ones(n_seq, n_seq), diagonal=1) * (-1e9)

    # Perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads,-1) # need to modify
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    
    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    generated_ids = []
    current_inputs = list(inputs)
    while len(generated_ids) < n_tokens_to_generate:
        temp = list(current_inputs)
        pos=len(current_inputs)-1
        
        for _ in range(K):#草稿
            draft_logits = gpt2(temp, draft_params, n_head=hparams_draft["n_head"])  
            draft_next_id =(np.argmax(draft_logits[-1]))  
            temp.append(int(draft_next_id))

#目标概率分布
        target_logits = gpt2(temp,target_params,n_head=hparams_target["n_head"])
        target_next_id = np.argmax(target_logits[-1])
        is_credible=True#连续几个草稿都正确
        
        for i in range(pos,pos+K):
            ps=draft_logits[i]
            qs=target_logits[i]
            pid=np.argmax(ps)
            qid=np.argmax(qs)
            generated_ids.append(int(qid))
            current_inputs.append(int(qid))

            if pid!=qid:
                is_credible=False
                break

        if is_credible:#奖励的一个
                generated_ids.append(int(target_next_id)) 
                current_inputs.append(int(target_next_id))
    generated_ids=generated_ids[:n_tokens_to_generate]
        

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5,draft_model_size:str ="124M", target_model_size: str = "1558M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    draft_encoder, hparams_draft, draft_params = load_encoder_hparams_and_params(draft_model_size, models_dir)
    target_encoder, hparams_target, target_params = load_encoder_hparams_and_params(target_model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = target_encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams_draft["n_ctx"]
    assert len(input_ids) + n_tokens_to_generate < hparams_target["n_ctx"]
    # generate output ids
    start = time.time()
    output_ids = greedy_speculative_generate(input_ids, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, 5)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = target_encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)