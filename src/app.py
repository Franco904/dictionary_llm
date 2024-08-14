from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


def define_dictionary_word(word: str):
    llm = CTransformers(
        model = 'models\llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type = 'llama',
        config = {
            'max_new_tokens': 256,
            'temperature': 0.01,
        },
    )

    template = """
        Devolva a definição de {word} no dicionário.
    """

    prompt = PromptTemplate(
        input_variables = ["word"],
        template = template,
    )

    response = llm(prompt.format(word=word))

    print(response)
