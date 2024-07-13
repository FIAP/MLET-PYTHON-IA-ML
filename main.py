import os
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Configurando o modelo Ollama
llm = Ollama(
    model="llama2",
    num_gpu=0,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

def gerar_insights_sobre_filmes(pergunta):
    # Cria o prompt com a pergunta do usuário
    prompt = f"Responda a seguinte pergunta sobre filmes: {pergunta}\n"
    prompt += "Por favor, forneça um resumo detalhado e quaisquer informações relevantes."

    # Usa o modelo Ollama para gerar a resposta
    insights = llm.invoke(prompt)
    return insights

def responder_pergunta(pergunta):
    # Gera a resposta usando o modelo Ollama
    resposta = gerar_insights_sobre_filmes(pergunta)
    return resposta

def main():
    # Pergunta do usuário
    pergunta = input("Sobre qual filme você deseja saber informações? ")
    resposta = responder_pergunta(pergunta)
    print(f"Resposta: {resposta}")

if __name__ == "__main__":
    main()
