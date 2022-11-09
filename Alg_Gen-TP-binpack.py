#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import bibliotek
import numpy as np
import time
import random
import matplotlib.pyplot as plt

import binpack_wykresy as bp
import sys


# In[2]:


# CONSTANTS
POP_SIZE = 20   # liczba osobnikow w populacji (tzw. rozmiar populacji)
ITER     = 100  # liczba iteracji -- liczba pokolen
MUT_PROB = 0.1  # prawdopodobieństwo mutacji
TOURNAMENT_SIZE = 5  # liczba osobników biorących udział w turnieju
STALL_GENERATIONS = 50

CONTAINERS_NUM = 4 # na ile podzielić sumę objętości paczek?
CONTAINERS_M = 4 # przez ile przemnożyć CONTAINERS_NUM?
MAX_VOLUME_RATIO = 0.7  # jaka część z sumy objętości powinna być dopuszczalna?


def prepare_problem(NG):
  items, max_capacity = bp.bpProblem(NG, CONTAINERS_NUM, np.random.random)
  return items, max_capacity


class Timer:
    def __init__(self):
        self.stoper = time.time()
    
    def next_result(self):
        res = time.time() - self.stoper
        self.stoper = time.time()
        return res


# In[3]:


# FUNKCJE POMOCNICZE Z WYKORZYSTANE Z WYKŁADU


def fitBinpack(individual, items = None, max_capacity = None):
  return - max(individual) - 1

def fitBinpackSupervisor(individual, items = None, max_capacity = None):
  return sum( [i+1 for i in individual] )

# FUNKCJE GLOWNE DLA POPULACJI

def get_init_popul(NG, items, max_capacity):  # poczatkowa populacja
  # początkową populację generujemy gorzej, niż to możliwe
  # bo inaczej AE w ogóle nie jest potrzebny
  # gdyż rozwiązania są od razu dobre
  return [bp.bpGentype4(items, max_capacity, CONTAINERS_NUM * CONTAINERS_M, np.random.random) for i in range(20)]

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
            if best[1] < i[1]:  # powinno być <
                best = i
    return best

# 2. bierzemy przy kazdych narodzinach dziecka ta sama pare rodzicow
def get_next_generation(popul, items, max_capacity):
    # ocena przystosowania każdego osobnika
    ev_pop = [ (i, fitBinpack(i, items, max_capacity)) for i in popul ]
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
        
        child1, child2 = bp.crossoverOnePoint(items, max_capacity, parent1[0], parent2[0], np.random.random)
        
        # co ciekawe, mutacja naprawdę wydłuża obliczenia
        if np.random.random() < MUT_PROB:
          child1 = bp.mutationOnePoint(items, max_capacity, child1, np.random.random)
        if np.random.random() < MUT_PROB:
          child2 = bp.mutationOnePoint(items, max_capacity, child2, np.random.random)
        
        children.append( child1 )
        if len(children) == POP_SIZE: #len(ev_pop):
            break
        children.append( child2 )
    #print('children', children[:3])
    
    #print('children:', children)
        
    return best, children


# FUNKCJA ZWRACAJACA KONOCOWA POPULACJE

def get_final_popul(NG, items, max_capacity):
    popul = get_init_popul(NG, items, max_capacity)
    
    stats = []
    for gen_num in range(ITER):
        best, popul = get_next_generation(popul, items, max_capacity)
        
        # aktualizacja statystyk
        if len(stats) == 0:
            stats.append( (gen_num, best) )
        else:
            if stats[-1][1][1] < best[1]:
                stats.append( (gen_num, best) )
            
        # dodatkowy warunek stopu
        if gen_num - stats[-1][0] > STALL_GENERATIONS:
            break
    
    # postępy AE - można zakomentować
    #for s in stats:
    #  print(s)
    
    return stats


def get_best_per(NG, items, max_capacity, with_print = True):
    '''zwraca najlepszego osobnika'''
    stats = get_final_popul(NG, items, max_capacity)

    #return stats[-1][1]
    return stats


# In[6]:


def simulation(NG):
    # zapamiętaj losowy stan generatora liczb pseudolosowych (PRNG)
    random_state = np.random.get_state()
    
    # ustawiamy PRNG na wartość stałą (początkową)
    np.random.seed(0)
    items, max_capacity = prepare_problem(NG)
    print('items', items)
    print('max_capacity', max_capacity)
    
    # odtwarzamy stan PRNG z początku
    np.random.set_state(random_state)
    
    timer = Timer()
    results = []
    
    for i in range(10):
      
        stats = get_best_per(NG, items, max_capacity)
        x_list, y_list = [], []
        for gen_num, (_, cont_num) in stats:
          x_list.append(gen_num)
          y_list.append(-cont_num)
          
        plt.plot(x_list, y_list)
        plt.xlabel('Nr pokolenia')
        plt.ylabel('Liczba kontenerów')
        plt.title('Wykres')
        plt.show()

        #best_sol, best_value = get_best_per(NG, items, max_capacity)
        #print(best_sol)
        #print(best_value)
        #print('--', i)
        results.append( timer.next_result() )
    
    avg_time = sum(results) / len(results)
    min_time = min(results)
    max_time = max(results)
    print('Średni czas: %.2f' % avg_time)
    return avg_time, min_time, max_time

def presentation(NG_list):
    #stoper = Timer()

    x_list, y_list = [], []
    for NG in NG_list:
        print('NG = ', NG)
        avg_time, min_time, max_time = simulation(NG)
        x_list.append(NG)
        y_list.append(avg_time)

    plt.plot(x_list, y_list)
    plt.xlabel('Liczba przedmiotów w plecaku -- rozmiar osobnika')
    plt.ylabel('Dlugosc trwania symulacji (w sekundach)')
    plt.title('Wykres')
    plt.show()
    #plt.savefig('wykres.png')

MY_NG = 20
presentation(list(range(MY_NG, MY_NG + 1)))


