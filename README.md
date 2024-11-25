# LLMGE

Evaluation of LLM graph reasoning ability

## Quick Start
Install dependencies:
```
pip install -r requirements.txt
```

Obtain one or more LLMs and modify chat.py:
- define function as `get_LLM_response(str) -> str` for getting single response from LLM
- modify `chat_dict` as `{'<LLM name>': <function>}`. This variable will be imported to get response

Try to run main_example.py to go through full process from data generation to result evaluation.
```
python main_example.py
```
**Attention**:
- You are not yet able to use 'few-shot' or 'CoT' unless shot prompts are generated. You will soon know how.
- The initial dataset is empty, in main_example.py, only a fraction of it is generated, and it is not the standard size.

---
full_generation.py provide shortcut for generating a full structured dataset. Run the following code by default will result in obtaining a smaller version for testing.
```
python full_generation.py
```

gen_prompt_example.py provide example of generating shot prompt and using it. It does not include dataset generation so please be sure that the dataset or the specific part of the dataset required exists.
```
python gen_prompt_example.py
```



