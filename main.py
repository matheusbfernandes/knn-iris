import numpy as np
import math
from sklearn.linear_model import LogisticRegression


class RBFN(object):
    def __init__(self, dados_x, dados_y, centroides, num_iteracoes=100, taxa_aprendizado=0.01, mostrar_custo=False):
        self.X = dados_x
        self.Y = dados_y
        self.num_iteracoes = num_iteracoes
        self.taxa_aprendizado = taxa_aprendizado
        self.centroides = centroides
        self.mostrar_custo = mostrar_custo
        self.W = np.random.randn(1, 2) * math.sqrt(2 / self.centroides.shape[1])
        self.b = np.zeros(1)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _funcao_binaria(z):
        return (z >= 0).astype(int)

    def _funcao_base_radial(self, x):
        # resultado = np.exp((-np.linalg.norm(x - self.centroides[0], axis=1, keepdims=True)) ** 2)
        # for i in range(1, self.centroides.shape[1]):
        #     resultado = np.concatenate((resultado, np.exp((- np.linalg.norm(x - self.centroides[i], axis=1, keepdims=True)) ** 2)), axis=1)
        resultado = np.empty((x.shape[0], self.centroides.shape[0]))
        for j in range(resultado.shape[1]):
            for i in range(resultado.shape[0]):
                resultado[i][j] = np.exp(-np.linalg.norm(x[i] - self.centroides[j]) ** 2)
        return resultado

    def _inicializar_beta(self):
        return np.ones(self.centroides.shape[0])

    def _forward_propagation(self, dados_entrada=None):
        if dados_entrada is None:
            dados_entrada = self.X

        resultado_forward = {
            "A0": dados_entrada
        }
        resultado_forward["Z"] = np.dot(self.W, resultado_forward["A0"].T) + self.b
        resultado_forward["A1"] = self._sigmoid(resultado_forward["Z"])

        return resultado_forward

    def predizer(self, dados_entrada):
        resultado_forward = self._forward_propagation(dados_entrada)
        return resultado_forward["A1"]

    def treinar(self):
        for i in range(self.num_iteracoes):
            resultado_forward = self._forward_propagation()
            # self.W = np.array([[-2.6, -2.6]])
            # self.b = np.array([2.84])
            self.W = self.W - self.taxa_aprendizado * np.dot((self.Y / - resultado_forward["A1"]) + ((1 - self.Y) / (1 - resultado_forward["A1"])), resultado_forward["A0"]) / self.X.shape[0]
            self.b = self.b - self.taxa_aprendizado * np.sum((self.Y / - resultado_forward["A1"]) + ((1 - self.Y) / (1 - resultado_forward["A1"]))) / self.X.shape[0]


def main():
    dados_x = np.array([[1, 1],
                        [1, 0],
                        [0, 1],
                        [0, 0]])
    dados_y = np.array([1, 0, 0, 0])
    centroides = np.array([[1, 1],
                           [1, 0],
                           [0, 1],
                           [0, 0]])
    rbfn = RBFN(dados_x, dados_y, centroides, num_iteracoes=100, taxa_aprendizado=0.0001, mostrar_custo=False)
    rbfn.treinar()
    teste = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [0, 0]])
    print(rbfn.predizer(teste))
    # print(rbfn._funcao_base_radial(dados_x))
    # lg = LogisticRegression()
    # lg.fit(rbfn._funcao_base_radial(dados_x), dados_y)
    # print(lg.score(rbfn._funcao_base_radial(dados_x), dados_y))


if __name__ == "__main__":
    main()
