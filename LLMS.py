from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
import time


class Llama2LLM:
    def __init__(self):
        self.llm = Llama(
            model_path='./models/codellama-7b.Q4_K_M.gguf', n_gpu_layers=1)

    def inference(self, input: str):
        output = self.llm(
            f'Q: {input} \n A:',
            max_tokens=64,
            stop=["Q:", "\n"]
        )
        return output['choices'][0]['text']


class Mistral7BInstructLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2")
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2")

    def inference(self, input: str):
        tokenized_inputs = self.tokenizer(input, return_tensors='pt')
        print(tokenized_inputs)
        output_ids = self.model.generate(
            tokenized_inputs.input_ids, max_length=64)
        output = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output[0]


# class Mistral7BInstructQuantizedLLM:
#     def __init__(self):
#         self.llm = Llama(model_path='./models/')


class LLMS:
    def __init__(self):
        self.llms = {
            'llama': Llama2LLM(),
            # 'mistral': Mistral7BInstructLLM()
        }

    def inference(self, input: str):
        result = {}
        for model_name, model in self.llms.items():
            start = time.perf_counter()
            model_output = model.inference(input)
            stop = time.perf_counter()
            result[model_name] = {
                'output': model_output,
                'time': stop - start
            }
        return result
