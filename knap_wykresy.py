#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import bibliotek
import numpy as np
import time
import random
import matplotlib.pyplot as plt


PLOTS_NUMBER = 10

# In[2]:


# CONSTANTS
POP_SIZE = 20   # liczba osobnikow w populacji (tzw. rozmiar populacji)
ITER     = 450  # liczba iteracji -- liczba pokolen
MUT_PROB = 0.75  # prawdopodobieństwo mutacji
TOURNAMENT_SIZE = 5  # liczba osobników biorących udział w turnieju
STALL_GENERATIONS = 50

MAX_VOLUME_RATIO = 0.7  # jaka część z sumy objętości powinna być dopuszczalna?

EXPERIMENT_REPETITIONS = 5000

def prepare_problem(NG):
    value  = np.random.random(size=NG) # losowopen e wartości (z numpy!)
    volume = np.random.random(size=NG) # losowe objetości 
    max_volume = sum(volume) * MAX_VOLUME_RATIO
    return volume, value, max_volume


class Timer:
    def __init__(self):
        self.stoper = time.time()
    
    def next_result(self):
        res = time.time() - self.stoper
        self.stoper = time.time()
        return res


# In[3]:


# FUNKCJE POMOCNICZE Z WYKORZYSTANE Z WYKŁADU

#losowy chromoson, z niego utworzona zostanie w kolejnych iteracjach populacja
def per(NG): # chromosom (osobnik, konfiguracja, przedmioty do spakowania, ...)
    return np.random.randint(2, size=NG) # wybrany losowo, parametr ng oznacza dlugość łańcucha znaków, gdzie 0 oznacze że nie 
                                         #zawiera przedmiotu a 1 że zawiera

fit = np.dot

# Returns total value, when total volume is not exceeded,
# and -1 otherwise.
def fitKnapsack(individual, value, volume, max_volume):
    total_vol = np.dot(individual, volume)
    if total_vol > max_volume:
        return -total_vol
    else:
        return np.dot(individual, value)

#ponieważ mamy do czynienia z problemem, gdzie wartości są binarne(0,1) to mutacja polega na zmianie z pewnym 
#prawdopodobieństwem losowego allelu w chromosomie
def mut(per, p): # per - osobnik, p - prawdopodobieństwo zajścia mutacji 
    if np.random.random() < p:
        ind=np.random.randint(len(per)) # losowo wybrany gen
        #print(ind)
        per[ind]=1-per[ind] # zmiana allelu: 0 <-> 1
    return per

# krzyżowanie
def cross2cut(papa, mama, cut):
    chp   = np.delete(papa, np.arange(cut, len(papa), 1))
    chm   = np.delete(mama, np.arange(0, cut, 1))
    child = np.concatenate((chp, chm), axis=0) # dziecko 1
    child = mut(child, MUT_PROB)
    return child

def cross2(papa, mama):
    cut = np.random.randint(1, len(papa))
    child1 = cross2cut(papa, mama, cut)
    child2 = cross2cut(mama, papa, cut)
    return child1, child2

# [0, 1, 1, | 0, 0]
# [1, 1, 0, | 0, 1]
# -----------------
# [0, 1, 1  | 0, 1]


# In[4]:


# FUNKCJE GLOWNE DLA POPULACJI

def get_init_popul(NG):  # poczatkowa populacja
    return [per(NG) for _ in range(POP_SIZE)]

def selTournament(ev_pop, tournament_size):
    # wybór losowych osobników do turnieju
    contestants = []
    for i in range(tournament_size):
        idx = np.random.randint(0, POP_SIZE) #len(ev_pop))
        contestants.append( ev_pop[idx] )
    
    #print('contestants:', contestants)
    
    # przystosowanie najlepszego osobnika w turnieju
    winner_fit = max([i[1] for i in contestants])
    
    # zwrócenie najlepszego osobnika z turnieju (zwycięzcy)
    return [i for i in contestants if i[1] == winner_fit][0]


def theSelection(ev_pop):
    parents = []
    
    # how many parents are needed?
    # (each pair of parents can produce two children)
    parents_needed = POP_SIZE # len(ev_pop)
    if parents_needed % 2 > 0:
        parents_needed += 1
    
    for i in range( parents_needed ):
        winner = selTournament(ev_pop, TOURNAMENT_SIZE)
        #print('winner is', winner)
        parents.append( winner )
    
    return parents

def bestEvaluated(ev_pop):
    best = None
    for i in ev_pop:
        if best is None:
            best = i
        else:
            if best[1] < i[1]:
                best = i
    return best

# 2. bierzemy przy kazdych narodzinach dziecka ta sama pare rodzicow
def get_next_generation(popul, value, volume, max_volume):
    # ocena przystosowania każdego osobnika
    ev_pop = [ (i, fitKnapsack(i, value, volume, max_volume)) for i in popul ]
    #print('ev_pop', ev_pop[:3])
    
    # dobrze będzie zapamiętać najlepsze rozwiązanie
    best = bestEvaluated(ev_pop)
    #print('best', best)
    
    # selekcja: wybór osobników rodzicielskich
    parents = theSelection(ev_pop)
    #print('parents', parents[:4])
    
    # krzyżowanie
    children = []
    while len(parents) > 0:
        parent1 = parents.pop()
        parent2 = parents.pop()
        child1, child2 = cross2(parent1[0], parent2[0])
        children.append( child1 )
        if len(children) == POP_SIZE: #len(ev_pop):
            break
        children.append( child2 )
    #print('children', children[:3])
    
    #print('children:', children)
        
    return best, children


    


# In[5]:


# FUNKCJA ZWRACAJACA KONOCOWA POPULACJE

def get_final_popul(NG, value, volume, max_volume):
    popul = get_init_popul(NG)
    #print(popul[:10])
    stats = []
    for gen_num in range(ITER):
        best, popul = get_next_generation(popul, value, volume, max_volume)
        
        # aktualizacja statystyk
        if len(stats) == 0:
            stats.append( (gen_num, best) )
        else:
            if stats[-1][1][1] < best[1]:
                stats.append( (gen_num, best) )
            
        # dodatkowy warunek stopu
        if gen_num - stats[-1][0] > STALL_GENERATIONS:
            break
    
    return stats


def get_best_per(NG, value, volume, max_volume, with_print = True):
    '''zwraca najlepszego osobnika'''
    stats = get_final_popul(NG, value, volume, max_volume)

    return stats[-1][1]

def get_best_per_progress(NG, value, volume, max_volume, with_print = True):
    '''zwraca najlepszego osobnika'''
    stats = get_final_popul(NG, value, volume, max_volume)

    return [(gen_num, val) for (gen_num, (_, val)) in stats]


# In[6]:

def simulationProgress(NG):
    # zapamiętaj losowy stan generatora liczb pseudolosowych (PRNG)
    random_state = np.random.get_state()
    
    # ustawiamy PRNG na wartość stałą (początkową)
    np.random.seed(0)
    volume, value, max_volume = prepare_problem(NG)
    print('volume', volume)
    print('value', value)
    print('max_volume', max_volume)
    
    # odtwarzamy stan PRNG z początku
    np.random.set_state(random_state)
    
    timer = Timer()
    results = []
 
    stats = get_best_per_progress(NG, value, volume, max_volume)
    #results.append( best_value )
    print(stats)
    return stats
    
    #print('Stopped after', iterations, 'iterations.')
    #min_val = min(results)
    #max_val = max(results)
    #return avg_val, min_val, max_val
    #avg_time = sum(results) / len(results)
    #min_time = min(results)
    #max_time = max(results)
    #print('Średni czas: %.2f' % avg_time)
    #return avg_time, min_time, max_time


def simulation(NG):
    # zapamiętaj losowy stan generatora liczb pseudolosowych (PRNG)
    random_state = np.random.get_state()
    
    # ustawiamy PRNG na wartość stałą (początkową)
    np.random.seed(0)
    volume, value, max_volume = prepare_problem(NG)
    print('volume', volume)
    print('value', value)
    print('max_volume', max_volume)
    
    # odtwarzamy stan PRNG z początku
    np.random.set_state(random_state)
    
    timer = Timer()
    results = []
    last_avg = None
    iterations = 500
    while True:
    
      for i in range(500):
      
        best_sol, best_value = get_best_per(NG, value, volume, max_volume)
        #results.append( best_value )
        results.append( timer.next_result() )
    
      avg_val = sum(results) / len(results)
      if last_avg is None:
        last_avg = avg_val
        iterations += 500
      else:
        if abs( avg_val - last_avg ) < 0.0005:
          break
        else:
          iterations += 500
          last_avg = avg_val
      
      print(last_avg, avg_val)

    print('Stopped after', iterations, 'iterations.')
    min_val = min(results)
    max_val = max(results)
    return avg_val, min_val, max_val
    #avg_time = sum(results) / len(results)
    #min_time = min(results)
    #max_time = max(results)
    #print('Średni czas: %.2f' % avg_time)
    #return avg_time, min_time, max_time

# mamy problem, z 20 przedmiotami
#   dla TEGO KONKRETNEGO PROBLEMU robimy eksperymenty z war. param. AG
#   -> uzyskujemy wartości parametrów, dla których wyniki nas zadowalają.

# 1) dla INNEGO problemu, nawet z 20 przedmiotami, nasze parametry nie muszą wcale dawać dobrych wyników
#    ALE: zakładamy, że nasze parametry są OK dla 20 przedmiotów.

# załóżmy: problem z 30 przedmiotami
# znowu robimy eksperymenty
# -> uzyskujemy wartości parametrów, dla których wyniki nas zadowalają.

# czy te "nowe" parametry są podobne/takie same, jak parametry "stare"?




def presentation(NG_list):
    global POP_SIZE, MUT_PROB
    #stoper = Timer()

    x_list, y_list = [], []
    for NG in NG_list:
        print('NG = ', NG)
        print('popsize:', POP_SIZE)
        print('mut_prob:', MUT_PROB)

        for i in range(PLOTS_NUMBER):
          stats = simulationProgress(NG)
          x_list, y_list = [], []
          for gen_num, val in stats:
            x_list.append(gen_num)
            y_list.append(val)
          
          plt.plot(x_list, y_list)
          plt.xlabel('Nr pokolenia')
          plt.ylabel('Wartość przedmiotów w plecaku')
          #plt.title('Wykres')
          plt.show()

        #print('avg =', avg, ', max =', mx, '. min =', mn)
        #x_list.append(NG)
        #y_list.append(avg)

    #plt.plot(x_list, y_list)
    #plt.xlabel('Liczba przedmiotów w plecaku -- rozmiar osobnika')
    #plt.ylabel('Dlugosc trwania symulacji (w sekundach)')
    #plt.title('Wykres')
    #plt.show()
    #plt.savefig('wykres.png')

MY_NG = 20
#presentation(list(range(MY_NG, MY_NG + 5)))
presentation([30])


# In[ ]:





# In[ ]:




