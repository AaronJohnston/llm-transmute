from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
import time


# class Llama2LLM:
#     def __init__(self):
#         self.llm = Llama(
#             model_path='./models/codellama-7b.Q4_0.gguf')

#     def inference(self, input: str):
#         output = self.llm(
#             f'Q: {input} \n A:',
#             max_tokens=64,
#             stop=["Q:", "\n"]
#         )
#         return output['choices'][0]['text']


# class Mistral7BInstructLLM:
#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "mistralai/Mistral-7B-Instruct-v0.2")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             "mistralai/Mistral-7B-Instruct-v0.2")

#     def inference(self, input: str):
#         tokenized_inputs = self.tokenizer(input, return_tensors='pt')
#         print(tokenized_inputs)
#         output_ids = self.model.generate(
#             tokenized_inputs.input_ids, max_length=64)
#         output = self.tokenizer.batch_decode(
#             output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

#         return output[0]


class Mistral7BInstructQuantizedQ3_K_MLLM:
    def __init__(self):
        self.llm = Llama(
            model_path='./models/mistral-7b-instruct-v0.2.Q3_K_M.gguf')

    def inference(self, original: str, modified: str):
        output = self.llm(
            # f'<s>[INST] {input} [/INST]',
            # f'{input}',
            mistral_transmute_prompt(original, modified),
            max_tokens=512,
            stop=['</s>']
        )
        return output['choices'][0]['text']


class Mistral7BInstructQuantizedQ5_K_MLLM:
    def __init__(self):
        self.llm = Llama(
            model_path='./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf')

    def inference(self, original: str, modified: str):
        output = self.llm(
            # f'<s>[INST] {input} [/INST]',
            # f'{input}',
            mistral_transmute_prompt(original, modified),
            max_tokens=512,
            stop=['</s>']
        )
        return output['choices'][0]['text']


class LLMS:
    def __init__(self):
        self.llms = {
            # 'llama': Llama2LLM(),
            'mistralQ3': Mistral7BInstructQuantizedQ3_K_MLLM(),
            'mistralQ5': Mistral7BInstructQuantizedQ5_K_MLLM(),
            # 'mistral': Mistral7BInstructLLM()
        }

    def inference(self, original: str, modified: str):
        result = {}
        for model_name, model in self.llms.items():
            start = time.perf_counter()
            model_output = model.inference(original, modified)
            stop = time.perf_counter()
            result[model_name] = {
                'output': model_output,
                'time': stop - start,
                'words': len(model_output.split())
            }
        return result


def mistral_transmute_prompt(original: str, modified: str):
    original_lines = original.split('\n')
    modified_lines = modified.split('\n')
    num_modified_lines = len(modified_lines)
    return '<s> [INSTR] Format these lines, returning only the new lines with no extra text: {} [/INSTR] {} </s> [INSTR] Good. Now format these lines the same way, returning only the new lines with no extra text: {} [\INSTR]'.format(original_lines[:num_modified_lines], modified_lines, original_lines[num_modified_lines:])
