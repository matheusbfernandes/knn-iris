import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNN(object):
    def __init__(self, dados_x, dados_y, k_visinhos):
        self.dados_x = dados_x
        self.dados_y = dados_y
        self.k_visinhos = k_visinhos

    # função responsável por retornar a distância euclidiana entre a instância e os dados de treino
    def _distancia_euclidiana(self, dado_teste):
        return np.linalg.norm(dado_teste - self.dados_x, axis=1, keepdims=True)

    # função responsável por retornar os k-vizinhos mais próximos, baseado na distância euclidiana
    def _get_vizinhos(self, dado_teste):
        vizinhos = []
        distancias_do_dado = self._distancia_euclidiana(dado_teste)
        for _ in range(self.k_visinhos):
            pos_menor = np.argmin(distancias_do_dado)
            vizinhos.append((np.amin(distancias_do_dado), pos_menor))
            distancias_do_dado = np.delete(distancias_do_dado, pos_menor, axis=0)
        return vizinhos

    def _get_classes(self, dados_reais=None):
        if dados_reais is None:
            dados_reais = self.dados_y
        return np.unique(dados_reais)

    # escolhe a classe da instância, baseado na quantidade dos k-vizinhos próximos daquela classe
    def _escolher_classe(self, vizinhos):
        classes = self._get_classes()
        classes[:] = 0
        for vizinho in vizinhos:
            posicao = vizinho[1]
            classe = self.dados_y[posicao]
            classes[classe] += 1
        return np.argmax(classes)

    # prediz qual classe uma determinada instância, ou instâncias, pertence
    def predizer(self, dado_teste):
        if dado_teste.shape[0] == self.dados_x.shape[1]:
            vizinhos = self._get_vizinhos(dado_teste)
            return self._escolher_classe(vizinhos)
        else:
            predito = np.empty(dado_teste.shape[0])
            for i in range(dado_teste.shape[0]):
                vizinhos = self._get_vizinhos(dado_teste[i])
                predito[i] = self._escolher_classe(vizinhos)
            return predito

    # mostra a taxa de acertos que o algoritmo atingiu (número que varia de 0 até 1)
    def taxa_acerto(self, dados_entrada, dados_saida):
        predito = self.predizer(dados_entrada)
        return np.sum((dados_saida == predito).astype(float)) / dados_saida.shape[0]

    def matriz_confusao(self, dados_reais, dados_preditos):
        classes = self._get_classes(dados_reais).shape[0]
        matriz_confusao = np.zeros((classes, classes))
        for i in range(dados_reais.shape[0]):
            matriz_confusao[dados_reais[i].astype(int)][dados_preditos[i].astype(int)] += 1

        return matriz_confusao.astype(int)


def main():
    # carregando o dataset Iris.csv
    dados_x, dados_y = load_iris(return_X_y=True)
    # dados_x, teste_x, dados_y, teste_y = train_test_split(dados_x, dados_y, test_size=0.2, random_state=42)

    knn = KNN(dados_x, dados_y, 4)
    print("Taxa de acerto: {:.2f}%".format(knn.taxa_acerto(dados_x, dados_y) * 100))
    print(knn.matriz_confusao(dados_y, knn.predizer(dados_x)))
    # print("Taxa de acerto: {:.2f}%".format(knn.taxa_acerto(teste_x[:, :2], teste_y) * 100))

    knn_sk = KNeighborsClassifier(4, algorithm='brute', p=2)
    knn_sk.fit(dados_x, dados_y)
    print(knn_sk.score(dados_x, dados_y) * 100)
    print(knn.matriz_confusao(dados_y, knn_sk.predict(dados_x)))
    # print(knn_sk.score(teste_x, teste_y) * 100)


if __name__ == "__main__":
    main()
