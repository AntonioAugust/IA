import math
import numpy as np


def funcao(x):
    y = 0.5*np.sin(3*x)+2*np.exp(-0.5*(x-10)**2)
    return y

def gera_vizinho(x, dx):
    xv = np.random.random() * (2*dx) + (x-dx) 
    return xv

#melhor resultados: solu_inicial: 10
#                   geravizinho( ,0.2)

    
temperatura_inicial = 100
temperatura_final = 1
num_iteracoes = 1000
taxa_resfriamento = 0.9

def SA(temperatura_inicial, temperatura_final, num_iteracoes, taxa_resfriamento):
    temperatura = temperatura_inicial
    solu_inicial = 8
    solu_atual = solu_inicial
    solu_otima = solu_atual
    
    while(temperatura > temperatura_final):
        for cont in range(num_iteracoes):
            solu_vizinho = gera_vizinho(solu_atual, 0.9)
            variacao_custo = funcao(solu_atual) - funcao(solu_vizinho)
            if variacao_custo < 0 :
                solu_atual = solu_vizinho
                
            else:
                n_random = np.random.random()
                if n_random < -(variacao_custo)/temperatura:
                    solu_atual = solu_vizinho
            if(funcao(solu_atual)>funcao(solu_otima)):
                solu_otima = solu_atual
        #print(temperatura)    
        temperatura = taxa_resfriamento * temperatura

    return solu_otima

melhor_solucao = SA(temperatura_inicial, temperatura_final, num_iteracoes, taxa_resfriamento)
print("Melhor solução(x)", melhor_solucao)
print("Melhor profundidade para plantar:", funcao(melhor_solucao))
