import torch
from transformers import AutoTokenizer
from einops._torch_specific import allow_ops_in_compiled_graph

import argparse

# class AndromedaEval:
#     def __init__(self, path, seed=42, device=None):
#         self.path = path
#         self.seed = seed

#         self.device = device

#         if self.device is None:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         set_seed(self.seed)

#         #tokenizer
#         self.tokenizer = AndromedaTokenizer

#         #model
#         self.model = Andromeda

#         #checkpoint
#         self.model.load_state_dict(torch.load(self.path))
#         self.model.eval()

#         #device
#         self.model = self.model.to(self.device)

#         #metrics
#         self.metrics = {}
#         self.reset_metrics()

#     def reset_metrics(self):
#         self.metrics = {
#             "generation_steps": None,
#             "time_forward": [],
#             "time_forward_average": None,

#             "memory_usages": [],
#             "memory_usage_average": None,
#             "time_end_to_end": None,

#             "throughput": None
#         }

#     def get_num_params(self):
#         num_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

#         return num_params

#     def generate(self, prompt, generation_steps=32):
#         #make sure all of the metrics reset at every generation
#         self.reset_metrics()

#         self.metrics["generation_steps"] = generation_steps

#         tokens = self.tokenizer.encode(prompt)
#         tokens_new = []

#         time_end_to_end = time.time()

#         #generation loop
#         for _ in range(generation_steps):
#             tokens_tensor = torch.tensor([tokens], device=self.device)

#             #forward pass
#             tracemalloc.start()

#             time_forward_0 = time.time()

#             logits = self.model(tokens_tensor, return_loss=False)[:, -1] # no loss takes the output of the last tokens

#             time_forward_1 = time.time()

#             _, memory_usage = tracemalloc.get_traced_memory()
#             tracemalloc.stop()

#             self.metrics["memory_usages"].append(memory_usage)

#             time_forward = time_forward_1 - time_forward_0
#             self.metrics["times_forward"].append(time_forward)

#             next_token = torch.armax(logits).item()

#             #save the newly generated token
#             tokens.append(next_token)
#             tokens_new.append(next_token)

#         time_end_to_end_1 = time.time()

#         time_end_to_end = time_end_to_end_1 - time_end_to_end_0
#         self.metrics["time_end_to_end"] = time_end_to_end

#         decoded = self.tokenizer.decode(tokens)

#         self.metrics["time_forward_average"] = np.mean(self.metrics["times_forward"])
#         self.metrics["memory_usage_average"] = np.mean(self.metrics["memory_usage"])

#         self.metrics['throughput'] = generation_steps / np.sum(self.metrics["times_forward"])

#         return tokens_new, decoded


# def main():
#     prompt = 'My name is'

#     andromeda = EvalAndromeda(path='checkpoints/step_44927_6656/pytorch_model.bin')

#     num_params = Andromeda.get_num_params()
#     print(f'The model has {num_params} parameters')

#     _, output = Andromeda.generate(prompt)

#     for metric, value in Andromeda.metrics.items():
#         print(f'{metric}: {value}\n')

#     print('\n')

#     print(output)


def main():
    allow_ops_in_compiled_graph()

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    parser = argparse.ArgumentParser(description="Generate text using Andromeda model")
    parser.add_argument("prompt", type=str, help="Text prompt to generate text")
    parser.add_argument(
        "--seq_len", type=int, default=256, help="Sequence length for generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--filter_thres", type=float, default=0.9, help="Filter threshold for sampling"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="andromeda-e-1",
        help="Model to use for generation",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help="Data type for the model: 'bf16', or 'fp32'",
    )

    args = parser.parse_args()

    dtype = torch.float32
    if args.dtype == "bf16":
        dtype = torch.bfloat16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # need to submit to torch hub
    model = torch.hub.load("apacai/andromeda", args.model).to(device).to(dtype)

    opt_model = torch.compile(model, backend="hidet")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    encoded_text = tokenizer(args.prompt, return_tensors="pt")

    output_tensor = opt_model.generate(
        seq_len=args.seq_len,
        prompt=encoded_text["input_ids"].to(device),
        temperature=args.temperature,
        filter_thres=args.filter_thres,
        pad_value=0.0,
        eos_token=tokenizer.eos_token_id,
        return_seq_without_prompt=False,
        use_tqdm=True,
    )

    decoded_output = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)

    return decoded_output


if __name__ == "__main__":
    generated_text = main()
    for text in generated_text:
        print(f"{text}")
