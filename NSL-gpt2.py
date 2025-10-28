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

def mha(x, attn, n_head, kv_cache, isfirst):  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    x_proj = linear(x, c_attn)  # [n_seq, 3*n_embd]
    # split q, k, v
    q, k, v = torch.chunk(x_proj, 3, dim=-1)  # each [n_seq, n_embd]

    # split into per-head tensors
    q_heads = q.chunk(n_head, dim=-1)
    k_heads = k.chunk(n_head, dim=-1)
    v_heads = v.chunk(n_head, dim=-1)

    n_seq = x.size(0)


    out_heads = []
    for i in range(n_head):
        qh = q_heads[i]      # [n_q, head_dim]
        kh = k_heads[i]      # [n_k_cur, head_dim]
        vh = v_heads[i]      #同k

        if (not isfirst) : 
            past_k, past_v = kv_cache[i]  # [n_past, head_dim], [n_past, head_dim]，取出历史kv
            past_k = past_k.to(kh.device)
            past_v = past_v.to(vh.device)
            #拼接历史kv与新token的kv
            k_comb = torch.cat([past_k, kh], dim=0)  # [n_past + n_k_cur, head_dim]
            v_comb = torch.cat([past_v, vh], dim=0)
            
            causal_mask = torch.zeros(qh.size(0), k_comb.size(0), dtype=x_proj.dtype, device=x_proj.device)
            # 更新kvcache
            kv_cache[i] = (k_comb, v_comb)
        else:
          
            k_comb = kh
            v_comb = vh
            causal_mask = torch.triu(torch.ones(n_seq, n_seq, device=kh.device), diagonal=1) * (-1e9)

            kv_cache[i] = (k_comb, v_comb)

        out_h = attention(qh, k_comb, v_comb, causal_mask)  # [n_q, head_dim]
        out_heads.append(out_h)

    # Merge heads and out projection
    x_out = torch.cat(out_heads, dim=-1)  # [n_q, n_embd]
    x_out = linear(x_out, c_proj)  # [n_q, n_embd] -> [n_q, n_embd]
    return x_out


def transformer_block(x, block, n_head,kv_cache,isfirst):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head,kv_cache=kv_cache,isfirst=isfirst)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head, kvcache_list, isfirst, pos_base=None):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']

    if (not isfirst) : 
        # 单个token增量输入
        x = wte[inputs] + wpe[pos_base]
        x = torch.Tensor(x)  # [1, n_embd]
    else:
        # 第一次调用
        x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
        x = torch.Tensor(x)


    for block, kv_cache in zip(blocks, kvcache_list):
        x = transformer_block(x, block, n_head=n_head, kv_cache=kv_cache, isfirst=isfirst)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    wte_t = torch.Tensor(wte)
    return x @ wte_t.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    isfirst = True
    # 初始化每一层的kvcache
    kvcache_list = [[None] * n_head for _ in range(len(params['blocks']))]

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        if isfirst:
            real_inputs = inputs  # 第一次
            pos_base = None
        else:
            real_inputs = [inputs[-1]]  # 单个token
            pos_base = len(inputs) - 1  # 这个token的位置

        logits = gpt2(real_inputs, params, n_head=n_head, kvcache_list=kvcache_list, isfirst=isfirst, pos_base=pos_base) 
        next_id = int(np.argmax(logits[-1]))
        inputs.append(next_id)
        isfirst = False

    # 清空
    for i in range(len(kvcache_list)):
        kvcache_list[i] = [None] * n_head

    return inputs[len(inputs) - n_tokens_to_generate :]  
    


def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        pass

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)