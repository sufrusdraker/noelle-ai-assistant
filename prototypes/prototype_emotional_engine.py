from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator


class EmotionalEngine:
    def __init__(self):
        """
        Inicializa o Banco de Memória Emocional (Valores de -10 a +10).
        Os eixos acumulam e reagem ao longo da interação com o usuário.
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.estado_emocional = {
            "conexao": 0,  # Fundação emocional / Afinidade
            "postura": 0,  # Nível de defesa / Orgulho (Tsundere)
            "estabilidade": 0  # Equilíbrio emocional / Nervosismo
        }

    def analisar_impacto_emocional(self, user_input):
        """
        Traduz o input, analisa a polaridade com o VADER e atualiza os eixos emocionais.
        """
        try:
            # Traduz do português para o inglês para otimizar a precisão do VADER
            frase_ingles = GoogleTranslator(source='pt', target='en').translate(user_input)
        except Exception:
            frase_ingles = user_input

        # Mede o sentimento (de -1.0 a +1.0)
        vs = self.analyzer.polarity_scores(frase_ingles)
        score = vs['compound']

        # Converte para uma variação de -4 a +4 na rodada
        variacao = int(score * 4)

        # 1. ATUALIZAÇÃO DO EIXO DE CONEXÃO
        # Se neutro, a conexão decai ou recupera lentamente em direção a zero
        if variacao == 0:
            if self.estado_emocional["conexao"] > 0:
                self.estado_emocional["conexao"] -= 1
            elif self.estado_emocional["conexao"] < 0:
                self.estado_emocional["conexao"] += 1
        else:
            self.estado_emocional["conexao"] += variacao

        # 2. LÓGICA COMPORTAMENTAL BASEADA NO ACÚMULO DA CONEXÃO
        if self.estado_emocional["conexao"] > 0:
            # Conexão positiva: Guarda baixa (postura cai) e timidez estabiliza
            self.estado_emocional["postura"] = -int(self.estado_emocional["conexao"] / 2)
            self.estado_emocional["estabilidade"] = -2
        else:
            # Conexão negativa (rejeição): Orgulho sobe como escudo defensivo
            self.estado_emocional["postura"] = abs(self.estado_emocional["conexao"])
            self.estado_emocional["estabilidade"] = self.estado_emocional["conexao"]

        # 3. TRAVA DE SEGURANÇA (Limites estritos de -10 a +10)
        for chave in self.estado_emocional:
            self.estado_emocional[chave] = max(-10, min(10, self.estado_emocional[chave]))

        return (
            self.estado_emocional["conexao"],
            self.estado_emocional["postura"],
            self.estado_emocional["estabilidade"]
        )


if __name__ == "__main__":
    # Instancia o motor emocional
    noelle_emotions = EmotionalEngine()

    print("Testando o sistema de Memória Emocional (Digite 'sair' para encerrar):")
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit"]:
            print("Encerrando simulador emocional.")
            break

        if user_input.strip():
            conexao, postura, estabilidade = noelle_emotions.analisar_impacto_emocional(user_input)
            print(f"Estado Atual -> Conexão: {conexao} | Postura: {postura} | Estabilidade: {estabilidade}\n")