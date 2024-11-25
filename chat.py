# this is just an example, please replace it with your own LLM api

def get_LLM_response(res_test: str) -> str:
    response = ''
    while len(response) <= 0 or 'error' in response.lower():
        try:
            response = llm_text_api(res_test)
        except Exception as e:
            print(f'[caught]: {e}')
            response = ''
    return response

# this variable is necessary
chat_dict = {
    'llm': get_LLM_response
}

if __name__ == '__main__':
    print(get_LLM_response('hi'))

