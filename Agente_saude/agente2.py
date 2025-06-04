import numpy as np 
import pandas as pd

data = pd.read_csv("Agente_saude/tabela_sintomas.csv", header=None)
data.drop(data.columns[[0]], axis=1, inplace=True)
data.drop([0], axis=0, inplace=True)
nome_dos_atributos = ["Febre","Tosse","Espirro","Dor de garganta","Coriza","Dor de cabeça"]

X = data[[1,2,3,4,5,6]]
Y = data[[7]]


#df = df[['coluna1', 'coluna2']]

def entropia(X, Y, indice_x):
    valores_possiveis = X[indice_x].unique()
    total_amostras = len(X)
    print(valores_possiveis)
    print(total_amostras)
    
    for valor in valores_possiveis:
        df = X[[indice_x]] == valor
        num_amostras = len(df) 
        Y_sub = Y[df[1]]
        #print(df)  
        print(Y_sub)     
    
entropia(X, Y, 1)  




def ganho_info(S, a):
    ...
    

class Node():
    def __init__(self):
        self.indice_do_atributo : int
        self.data : int
        self.filho_left : Node 
        self.filho_right : Node
        self.e_folha  : bool
        self.classe : str
        



class decision_tree():

    def __init__(self, novo):
        self.root : Node = Node(None, None)
        
    def fit(X, Y): # X entradas, Y classes
        ...
    
    def build_tree(X, Y):
        ...
