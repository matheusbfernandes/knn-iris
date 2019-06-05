import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class KNN(object):
    def __init__(self, treino_x, treino_y, k_vizinhos):
        self.treino_x = treino_x
        self.treino_y = treino_y
        self.k_vizinhos = k_vizinhos

    # função responsável por retornar a distância euclidiana entre a instância e os dados de treino
    def _distancia_euclidiana(self, valor_exemplo):
        return np.linalg.norm(valor_exemplo - self.treino_x, axis=1)

    # função responsável por retornar os k-vizinhos mais próximos, baseado na distância euclidiana
    def _get_vizinhos(self, valor_exemplo):
        distancias_do_dado = self._distancia_euclidiana(valor_exemplo)
        vizinhos = np.concatenate((distancias_do_dado.reshape(distancias_do_dado.shape[0], 1), self.treino_y.reshape(self.treino_y.shape[0], 1)), axis=1)
        vizinhos = vizinhos[vizinhos[:, 0].argsort(kind="heapsort")]

        return vizinhos[:self.k_vizinhos, :]

    def _get_classes(self, dados=None):
        if dados is None:
            dados = self.treino_y
        return np.unique(dados)

    # faz a votação para escolher a classe da instância
    def _escolher_classe(self, vizinhos):
        classes = self._get_classes()
        classes[:] = 0
        for vizinho in vizinhos:
            classe = int(vizinho[1])
            classes[classe] += 1

        return np.argmax(classes)

    # prediz qual classe uma determinada instância, ou instâncias, pertence
    # pode receber como parâmetro um dado individual, ou um numpy array
    def predizer(self, valor_exemplo):
        if valor_exemplo.shape[0] == self.treino_x.shape[1]:
            vizinhos = self._get_vizinhos(valor_exemplo)
            return self._escolher_classe(vizinhos)
        else:
            predito = np.empty(valor_exemplo.shape[0])
            for i in range(valor_exemplo.shape[0]):
                vizinhos = self._get_vizinhos(valor_exemplo[i])
                predito[i] = self._escolher_classe(vizinhos)
            return predito

    # recebe como parâmetros dois numpy arrays e gera as métricas de avaliação para o dataset,
    # sendo elas: Precison, Recall, F1-Score e Accuracy
    def metricas_avaliacao(self, entrada_x, saida_y):
        classes = self._get_classes()
        predito = self.predizer(entrada_x)
        cm = self.matriz_confusao(saida_y, predito)
        diagonal = np.diag(cm)

        resultado = "+--------------------------------------+\n"
        resultado += "|  -  | Precisão |  Recall  | F1-Score |\n"
        for i in range(classes.shape[0]):
            precision = diagonal[i] / np.sum(cm[:, i])
            recall = diagonal[i] / np.sum(np.where(saida_y == classes[i])[0].shape[0])
            f1_score = 2 * precision * recall / (precision + recall)
            resultado += "|{:^5d}|{:^10.2f}|{:^10.2f}|{:^10.2f}|\n".format(int(classes[i]), precision, recall, f1_score)
        resultado += "+--------------------------------------+\n"
        resultado += "Acurácia: {:.2%}".format(np.sum((saida_y == predito).astype(float)) / saida_y.shape[0])

        return resultado

    # gera a matriz de confusão baseado em dois parâmetros:
    # dados_reais -> numpy array que indica os valores verdadeiros dos labels
    # dados_preditos -> numpy array que indica os valores profetizados pelo algoritmo
    def matriz_confusao(self, dados_reais, dados_preditos):
        classes = self._get_classes(dados_reais).shape[0]
        matriz_confusao = np.zeros((classes, classes))
        for i in range(dados_reais.shape[0]):
            matriz_confusao[dados_reais[i].astype(int)][dados_preditos[i].astype(int)] += 1

        return matriz_confusao.astype(int)


def main():
    # carregando o dataset Iris.csv
    dados_x, dados_y = load_iris(return_X_y=True)
    treino_x, teste_x, treino_y, teste_y = train_test_split(dados_x, dados_y, test_size=0.2, random_state=42)

    # iniciando o algoritmo
    knn = KNN(treino_x, treino_y, 7)

    # realizando os cálculos no training set
    print("Resultados obtidos do training set:")
    print(knn.metricas_avaliacao(treino_x, treino_y))
    print("\nMatriz de confusão:\n{:s}".format(str(knn.matriz_confusao(treino_y, knn.predizer(treino_x)))))

    # realizando os cálculos no test set
    print("\n\nResultados obtidos do test set:")
    print(knn.metricas_avaliacao(teste_x, teste_y))
    print("\nMatriz de confusão:\n{:s}".format(str(knn.matriz_confusao(teste_y, knn.predizer(teste_x)))))


if __name__ == "__main__":
    main()
