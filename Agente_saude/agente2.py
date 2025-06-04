
import numpy as np 
import pandas as pd
from io import StringIO
from math import log2

dados_csv = """Paciente,Febre,Tosse,Espirro,Dor de garganta,Coriza,Dor de cabeça,Doença
1,comum,comum,raro,as vezes,raro,as vezes,covid19
2,raro,as vezes,comum,comum,comum,raro,resfriado
3,comum,comum,raro,as vezes,as vezes,comum,gripe
4,comum,comum,raro,as vezes,as vezes,comum,gripe
5,comum,comum,raro,as vezes,raro,as vezes,covid19
6,raro,as vezes,comum,comum,comum,raro,resfriado
7,comum,comum,raro,as vezes,raro,as vezes,covid19
8,comum,comum,raro,as vezes,as vezes,comum,gripe"""

data = pd.read_csv(StringIO(dados_csv))
data.drop(data.columns[[0]], axis=1, inplace=True)
#data.drop([0], axis=0, inplace=True)
nome_dos_atributos = ["Febre","Tosse","Espirro","Dor de garganta","Coriza","Dor de cabeça"]

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]


def calcular_entropia(coluna_y):
    valores, contagens = np.unique(coluna_y, return_counts=True)
    total = len(coluna_y)
    #print("Valores possiveis para o atributo{indice_x}:", valores)
    #print("Total de amostras em um valor:", contagens)
    #print("total de amostras:", total)
    entropia = 0
    
    for cont in contagens:
        prob = cont / total
        entropia -= prob * log2(prob)
    
    return entropia


def entropia_condicional(X_coluna, Y):
    valores_atributo = X_coluna.unique()
    total_amostras = len(X_coluna)
    entropia_total = 0

    for valor in valores_atributo:
        indices = X_coluna == valor
        subset_Y = Y[indices]
        peso = len(subset_Y) / total_amostras
        entropia_subset = calcular_entropia(subset_Y)
        entropia_total += peso * entropia_subset

    return entropia_total


def ganho_info(X, Y, nome_atributo):
    entropia_inicial = calcular_entropia(Y)
    entropia_atributo = entropia_condicional(X[nome_atributo], Y)
    ganho = entropia_inicial - entropia_atributo
    return ganho
    

class Node():
    def __init__(self, atributo=None, filhos=None, classe=None):
        self.atributo = atributo
        self.filhos = filhos or {}
        self.classe = classe
    def is_leaf(self):
        return self.classe is not None
        

class decision_tree():

    def __init__(self):
        self.root = None
        
    def fit(self, X, Y): # X entradas, Y classes
        self.root = self.build_tree(X,Y)
    
    def build_tree(self, X, Y):
        # Caso 1: todos os exemplos têm a mesma classe
        if len(set(Y)) == 1:
            return Node(classe=Y.iloc[0])

        # Caso 2: não há mais atributos para dividir
        if X.empty:
            classe_mais_comum = Y.mode()[0] #Y.mode() calcula a moda da série Y, ou seja, o valor que mais aparece (a classe mais frequente).
            return Node(classe=classe_mais_comum)

        # Escolher o melhor atributo pelo ganho de informação
        ganhos = {col: ganho_info(X, Y, col) for col in X.columns}
        print("ganhos dos atributos", ganhos)
        melhor_atributo = max(ganhos, key=ganhos.get)

        # Criar o nó e dividir pelos valores do melhor atributo
        raiz = Node(atributo=melhor_atributo)

        for valor in X[melhor_atributo].unique():
            indices = X[melhor_atributo] == valor
            X_sub = X[indices].drop(columns=[melhor_atributo])
            Y_sub = Y[indices]

            if Y_sub.empty:
                classe_mais_comum = Y.mode()[0]
                raiz.filhos[valor] = Node(classe=classe_mais_comum)
            else:
                raiz.filhos[valor] = self.build_tree(X_sub, Y_sub)

        return raiz

    def predict(self, X):
        return [self._predict_linha(self.root, linha) for _, linha in X.iterrows()]
    
    def _predict_linha(self, node, linha):
        if node.is_leaf():
            return node.classe
        
        valor = linha[node.atributo]
        if valor in node.filhos:
            return self._predict_linha(node.filhos[valor], linha)
        else:
            return "classe_desconhecidsa"
        

arvore = decision_tree()
arvore.fit(X, Y)

# Testando com os próprios dados
predicoes = arvore.predict(X)
for i, (real, pred) in enumerate(zip(Y, predicoes), start=1):
    print(f"Paciente {i}: Real = {real}, Predito = {pred}")

def predict_usuario(arvore, atributos):
    print("Digite os sintomas com os seguintes valores possíveis: comum, raro, as vezes")
    entrada = {}

    for atributo in atributos:
        valor = input(f"{atributo}: ").strip().lower()
        while valor not in ["comum", "raro", "as vezes"]:
            print("Valor inválido. Use apenas: comum, raro ou as vezes.")
            valor = input(f"{atributo}: ").strip().lower()
        entrada[atributo] = valor

    linha = pd.Series(entrada)
    classe_predita = arvore._predict_linha(arvore.root, linha)
    print(f"\nA classe predita para os sintomas fornecidos é: {classe_predita}")

predict_usuario(arvore, nome_dos_atributos)
