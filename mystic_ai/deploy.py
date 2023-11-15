from pipeline.cloud import environments
from pipeline.cloud.pipelines import upload_pipeline
from pipeline.cloud.compute_requirements import Accelerator
from pipeline import entity, pipe, Pipeline, Variable
from pipeline.configuration import current_configuration

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

current_configuration.set_debug_mode(True)

base_model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
ft_model_id = 'textminr/mistral-7b-4bit-tl'

tl_prompt = '### Topic Words: {}. ### Topic Label:'

@entity
class MistralTLPipeline:

  @pipe(on_startup = True, run_once = True)
  def load_model(self) -> None:
    bnb_config = BitsAndBytesConfig(
      load_in_4bit = True,
      bnb_4bit_use_double_quant = True,
      bnb_4bit_quant_type = "nf4",
      bnb_4bit_compute_dtype = torch.bfloat16
    )

    self.base_model = AutoModelForCausalLM.from_pretrained(
      base_model_id,
      quantization_config = bnb_config,
      trust_remote_code = True
    )

    self.tokenizer = AutoTokenizer.from_pretrained(
      base_model_id,
      add_bos_token = True,
      torch_dtype = torch.float16,
      trust_remote_code = True
    )

    self.model = PeftModel.from_pretrained(self.base_model, ft_model_id)
    self.model.eval()

  @pipe
  def inference(self, words: str) -> str:
    prompt = tl_prompt.format(words)
    
    model_input = self.tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
      return self.tokenizer.decode(
        self.model.generate(**model_input, max_new_tokens = 60, repetition_penalty = 1.15)
          [0][model_input['input_ids'].shape[1]:],
        skip_special_tokens = True
      )
    
with Pipeline() as builder:
  words = Variable(str, title = 'Topic Words')

  _pipeline = MistralTLPipeline()
  _pipeline.load_model()
  out = _pipeline.inference(words)

  builder.output(out)

tl_pipeline = builder.get_pipeline()

tl_env = environments.create_environment(
  'asuender/mistral_tl',
  python_requirements = [
    'bitsandbytes==0.41.0',
    'torch==2.1.0',
    'git+https://github.com/huggingface/transformers.git',
    'git+https://github.com/huggingface/peft.git',
    'git+https://github.com/huggingface/accelerate.git'
  ]
)

upload_pipeline(
  tl_pipeline,
  'asuender/textminr_mistral_tl',
  'asuender/mistral_tl',
  minimum_cache_number = 1,
  required_gpu_vram_mb = 8_000,
  accelerators = [
    Accelerator.nvidia_t4
  ]
)