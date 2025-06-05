import numpy as np

def funcao(x):
    y = 0.5*np.sin(3*x)+2*np.exp(-0.5*(x-10)**2)
    return y

def gera_vizinho(x, dx):
    xv = np.clip(np.random.random() * (2*dx) + (x - dx), 0, 20)
    return xv
 
temperatura_inicial = 100
temperatura_final = 1
num_iteracoes = 1000
taxa_resfriamento = 0.9
solu_inicial = np.random.uniform(0, 20)

def SA(temperatura_inicial, temperatura_final, num_iteracoes, taxa_resfriamento, solu_inicial):
    temperatura = temperatura_inicial
    solu_inicial = solu_inicial
    
    solu_atual = solu_inicial
    solu_otima = solu_atual
    
    while(temperatura > temperatura_final):
        for cont in range(num_iteracoes):
            solu_vizinho = gera_vizinho(solu_atual, 0.2)
            variacao_custo = funcao(solu_atual) - funcao(solu_vizinho)
            if variacao_custo < 0 :
                solu_atual = solu_vizinho
                
            else:
                n_random = np.random.random()
                if n_random < np.exp(-(variacao_custo)/temperatura):
                    solu_atual = solu_vizinho
            if(funcao(solu_atual)>funcao(solu_otima)):
                solu_otima = solu_atual  
        temperatura = taxa_resfriamento * temperatura
        #print(f"Temp: {temperatura:.2f}, Melhor X: {solu_otima:.4f}, Valor: {funcao(solu_otima):.4f}")

    return solu_otima

melhor_solucao = SA(temperatura_inicial, temperatura_final, num_iteracoes, taxa_resfriamento, solu_inicial)
print(f"Melhor profundidade para plantar: {melhor_solucao:.2f} metros")
print(f"Valor da função utilidade nessa profundidade: {funcao(melhor_solucao):.4f}")
