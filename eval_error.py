from transformers import AutoModelForCausalLM , AutoTokenizer
import fnmatch
import argparse
import torch

device = torch.device("cuda:0")

def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], num_fewshot=0, add_special_tokens=False):
    return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
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

    if args.model[0] != '/':
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("model loading complete")

    model = model.to(device)

    task_list = ["hellaswag", "winogrande", "arc_easy", "arc_challenge"]
    num_shot = 0
    results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot)
    print("********************************")
    print("zero_shot evaluation results")
    print(results)

if __name__ == '__main__':
    main()