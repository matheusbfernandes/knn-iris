import numpy as np
from sklearn.cluster import KMeans


class RBFN(object):
    def __init__(self, dados_x, dados_y, centros, taxa_aprendizado=0.1, num_iteracoes=100):
        self.dados_x = dados_x
        self.dados_y = dados_y
        self.centros = centros
        self.W = np.zeros(centros.shape[0] + 1)
        self.taxa_aprendizado = taxa_aprendizado
        self.num_iteracoes = num_iteracoes
        self.sigma = self._calcular_sigma()

    @staticmethod
    def _funcao_binaria(z):
        return (z >= 0).astype(int)

    def _calcular_sigma(self):
        temp = np.empty(self.centros.shape[0])
        for i in range(temp.shape[0]):
            temp[i] = np.sum(np.linalg.norm(self.dados_x - self.centros[i], axis=0)) / self.dados_x.shape[0]
        return temp

    def _funcao_base_radial(self, x):
        resultado = np.empty((x.shape[0], self.centros.shape[0]))
        for j in range(resultado.shape[1]):
            for i in range(resultado.shape[0]):
                resultado[i][j] = np.exp(-1 / ((2 * self.sigma[j]) ** 2) * np.linalg.norm(x[i] - self.centros[j]) ** 2)
        return resultado

    def predizer(self, X=None):
        if X is not None:
            X = self._funcao_base_radial(X)
            X = np.insert(X, 0, 1, axis=1)
        else:
            X = self.dados_x

        z = np.dot(self.W, X.T)
        return self._funcao_binaria(z)

    def treinar(self):
        X = self._funcao_base_radial(self.dados_x)
        X = np.insert(X, 0, 1, axis=1)
        self.dados_x = X

        for _ in range(self.num_iteracoes):
            y_chapeu = self.predizer()
            self.W = self.W + self.taxa_aprendizado * (np.dot((self.dados_y - y_chapeu), X) / X.shape[0])


if __name__ == '__main__':
    dados_entrada = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    dados_saida = np.array([0, 1, 1, 0])

    kmeans = KMeans(n_clusters=4, precompute_distances=True).fit(dados_entrada)
    centroides = kmeans.cluster_centers_

    rbfn = RBFN(dados_entrada, dados_saida, centroides, taxa_aprendizado=1.35, num_iteracoes=50)
    rbfn.treinar()

    dados_teste = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    print("Testes realizados com as entradas:\n{:s}\n".format(str(dados_teste)))
    print("Utilizando os centros:\n{:s}\n".format(str(centroides)))
    print("Geraram as seguintes saídas:\n{:s}".format(str(np.reshape(rbfn.predizer(dados_teste), (-1, 1)))))
