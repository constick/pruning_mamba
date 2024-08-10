from transformers import AutoModelForCausalLM , AutoTokenizer
import numpy as np
import argparse
import torch

def model_weights_dict(model, dev="cuda:0"):
    res_dict = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            res_dict[n] = m.weight.to(device=dev)
        elif isinstance(m, torch.nn.Linear):
            res_dict[n] = m.weight.to(device=dev)

    return res_dict

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--full_model', type=str, help='full_model_size')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search", "full"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    full_model = AutoModelForCausalLM.from_pretrained(args.full_model, trust_remote_code=True)
    pruned_model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    full_model_weight_dict = model_weights_dict(full_model)
    pruned_model_weight_dict = model_weights_dict(pruned_model)

    mean_mse = 0
    mean_sqnr = 0

    for name, tensor in full_model_weight_dict.items():
        full_model_weights = tensor.flatten()
        pruned_model_weights = pruned_model_weight_dict[name].flatten()

        mse = torch.mean((full_model_weights-pruned_model_weights) ** 2)
        tensor_norm = torch.mean(full_model_weights**2)
        if mse.item() > 0.0:
            pruning_sqnr = 10 * np.log10(tensor_norm.item() / mse.item())
        else:
            pruning_sqnr = 0

        print('layer:', name, 'mse=', mse.item(), 'sqnr=', pruning_sqnr)

        mean_mse += mse.item()
        mean_sqnr += pruning_sqnr

    mean_mse = mean_mse/(len(full_model.model.layers)*4)
    mean_sqnr = mean_sqnr/(len(full_model.model.layers)*4)

    print('mse=', mean_mse, 'sqnr=', mean_sqnr)

if __name__ == '__main__':
    main()