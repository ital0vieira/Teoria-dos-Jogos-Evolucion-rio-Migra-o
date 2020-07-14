# Teoria-dos-Jogos-Evolucion-rio-Migra-o
DINÂMICA DE MIGRAÇÕES APLICADAS A TEORIA DOS JOGOS EVOLUCIONÁRIOS 

import numpy as np  # biblioteca numpy, para computação científica, utiliza o apelido np
import random
import matplotlib.pyplot as plt
import time
from numba import jit
start_time=time.time()
#Parâmetros da simulação:
L=100 #tamanho da rede
amostras=10
total_passos = 100
passos_media=total_passos-100
k = 0.1
R=[1,1] #Detalhe importante, cada rank dos payoffs se refere a uma cidade específica
P=[0,0]
T=[0.89,1.05]
S=[0,0]
p_mig=0#taxa de migração
n_mig=0 #número de imigrantes por MCS
total_jog = L*L

@jit(nopython=True)
def inicia_vizinhos(viz):
    # Inicialização das estratégias e definição de vizinhos
    for jogador_atual in range(0, total_jog):  # Lembrete, for in range vai até total_jog-1!!
        viz[jogador_atual, 1] = jogador_atual + 1  # Vizinho direito=1
        viz[jogador_atual, 3] = jogador_atual - 1  # Vizinho esquerdo=3
        viz[jogador_atual, 2] = jogador_atual + L  # Vizinho de baixo=2
        viz[jogador_atual, 0] = jogador_atual - L  # Vizinho de cima=0
    # definindo condições de contorno periódicas na rede quadrada
    for jogador_atual in range(0, total_jog):
        if ((jogador_atual - L + 1) % L == 0):  # definindo coluna direita
            viz[jogador_atual, 1] = jogador_atual + 1 - L  # definindo vizinho direito
        if (jogador_atual % L == 0 or jogador_atual == 0):  # definindo coluna esquerda
            viz[jogador_atual, 3] = jogador_atual + L - 1  # definindo vizinho esquerdo
        if (jogador_atual < L and jogador_atual >= 0):
            viz[jogador_atual, 0] = jogador_atual + L * L - L
        if (jogador_atual < L * L and jogador_atual >= L * L - L):
            viz[jogador_atual, 2] = jogador_atual - (L * L - L)

@jit(nopython=True)
def inicia_estrategias(estrategia):
    # Inicialização das estratégias
    for jogador_atual in range(0, total_jog):  # Lembrete, for in range vai até total_jog-1!!
        estrategia[jogador_atual] = random.randint(0, 1)  # dEstado inicial do jogador como aleatório

@jit(nopython=True)
def calcula_payoff(sitio,rede,estrategia):
    payoff_sitio = 0
    for j in range(0, 4):  # soma-se ao payoff o valor do jogo com vizinho para cada vizinho de 0 a 3
        payoff_sitio += matriz_payoff[rede][estrategia[sitio]][estrategia[viz[sitio, j]]]
    return payoff_sitio

@jit(nopython=True)
def atualiza_total_estrat(rede,estrategia):
    for i in range(0, total_jog):
        atual=random.randint(0, total_jog-1)
        sorteio = random.randint(0, 3)      #sorteio de um vizinho aleatório de 0 a 3
        vizsorteado=viz[atual,sorteio]
        pay_atual=calcula_payoff(atual, rede, estrategia)
        pay_viz  =calcula_payoff(vizsorteado, rede, estrategia)
        prob = random.random()                                          #probabilidade aleatória
        varia_payoff=pay_viz-pay_atual  #diferença de payoff do sítio e vizinho
        chance_muda = 1/( 1+np.exp( -varia_payoff/k ) )                    #Probabilidade de fermi
        if (prob<chance_muda):
            estrategia[atual] = estrategia[vizsorteado]  # mudança da estratégia do sítio central

@jit(nopython=True)
def migracao(estrategia,estrategia1):
    for i in range(0,n_mig):
        prob = random.random()
        if (prob<p_mig):
            imig0 = random.randint(0, total_jog - 1)
            imig1 = random.randint(0, total_jog - 1)
            aux=estrategia[imig0]
            estrategia[imig0]=estrategia1[imig1]  # mudança da estratégia do sítio central
            estrategia1[imig1]=aux  # mudança da estratégia do sítio central

#*******************************************************************************
#Definição de variáveis do jogo
#*******************************************************************************
matriz_payoff = np.zeros((2, 2,2))
rhoT=np.zeros(2)
sigmaT=np.zeros(2)
estrategia = np.zeros((total_jog), dtype=int)
estrategia1 = np.zeros((total_jog), dtype=int)
#Definição de vizinhos
viz=np.zeros( (total_jog,4),dtype=int)      #matriz contendo os vizinhos 0=cima,1=direita,2=baixo,3=esquerda
#Definição de variáveis de dados e estatísticas
arquivo_escrito= open("desertores.dat","w+")
cooperadores_t = np.zeros((2,total_passos))
cooperadores_medios_t = np.zeros((2,total_passos))
#0 Copera 1 Deserta
for i in range(0,2):     #Detalhe importante, cada rank inicial dos payoffs se refere a uma cidade específica
    matriz_payoff[i][0][0] = R[i] # Recompensa
    matriz_payoff[i][1][0] = T[i] # Tentação
    matriz_payoff[i][1][1] = P[i] # Punição
    matriz_payoff[i][0][1] = S[i] # Sucker
#************************************** início*************************************************
#**************************************************************************************************
inicia_vizinhos(viz)
for amo in range(0,amostras):       #loop de amostras
    inicia_estrategias(estrategia)
    inicia_estrategias(estrategia1)
    #Começo da simulação de Monte-Carlo
    #****************************************************************
    for passo_atual in range (0, total_passos):
        #Antes de qualquer alteração, calcular todas estatísticas da população no passo atual
        cooperadores_t[0,passo_atual] = 1-(sum(estrategia)/total_jog)   #Rede0
        cooperadores_t[1,passo_atual] = 1-(sum(estrategia1)/total_jog)  #Rede1
        for i in range (0,2):
            cooperadores_medios_t[i,passo_atual] += cooperadores_t[i,passo_atual]
        # Etapa de atualização da estratégia
        # ****************************************************************
        atualiza_total_estrat(0,estrategia) #atualiza cada rede individualmente
        atualiza_total_estrat(1,estrategia1)
        #Etapa de migração
        # ****************************************************************
        migracao(estrategia, estrategia1)
        #Fim do Monte-Carlo para uma amostra com "total_passos" passos de tempo
    plt.figure(1)
    for i in range(0,2):
        plt.subplot(2,1,i+1) #rows, coluns, index
        plt.plot(cooperadores_t[i,:])
    #Fim do loop para todas amostras
cooperadores_medios_t=cooperadores_medios_t/amostras
for i in range(0, total_passos):
    arquivo_escrito.write("%d %f %f \n" % (i, cooperadores_medios_t[0, i], cooperadores_medios_t[1, i]))
arquivo_escrito.write("\n")

#Generating figures:
plt.figure(1)
plt.suptitle('Cooperadores, por amostra')
for i in range(0, 2):
    plt.subplot(2,1,i+1)
    plt.ylabel(r'$\rho$') # r transforma a string e mraw, permitindo comandos latex
plt.xlabel('Número de Passos')
plt.figure(2)
plt.suptitle('Evolução média da cooperação')
for i in range(0, 2):
    plt.subplot(2,1,i+1)
    plt.ylabel(r'$<\rho >$')
    plt.plot(cooperadores_medios_t[i,:])
plt.xlabel('Número de Passos')
plt.figure(3)
plt.suptitle('Histograma de distribuição final de cooperadores')
for i in range(0, 2):
    plt.subplot(2,1,i+1)
    plt.xlim(-0.1, 1.1)
    plt.hist(cooperadores_medios_t[i,passos_media:total_passos])
    plt.ylabel(r'$P(\rho)$')
plt.xlabel(r'$\rho$')

for i in range(0,2):
    rhoT[i]=np.average(cooperadores_medios_t[i,passos_media:total_passos])   #Média dos ultimos tempos para rho p/ todas amostras
    sigmaT[i]=np.std(cooperadores_medios_t[i,passos_media:total_passos])     #desvio padrão destes
print('<C1>= %f +- %f  <C2>= %f +- %f ' % (rhoT[0],sigmaT[0],rhoT[1],sigmaT[1]) )
print("----- %f seconds or %f minutes-----" % (time.time()-start_time , (time.time()-start_time)/60.0  ))
arquivo_escrito.close()
plt.show()
