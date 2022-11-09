import math
import random

def rnd_int(rnd_func, mn, mx):
  threshold = 1 / (mx - mn + 1)
  value = rnd_func()
  return mn + math.floor(value / threshold)

# losowy problem kontenerowy
#   n - liczba przedmiotów
#   k - orientacyjna liczba kontenerów
#   rnd_func - funkcja zwracająca wartość pseudolosową [0; 1)
#
# Zwraca: listę przedmiotów, pojemność kontenera
def bpProblem(n, k, rnd_func):
  items = [rnd_func() for i in range(n)]
  container_capacity = sum(items) / k
  return items, container_capacity

# to samo, co v3, ale "równomiernie" rozkłada wszystkie grupy po genotypie
def bpGentype4(items, capacity, k, rnd_func):
  gtype = []
  groups = {}
  for i in range(k):
    groups[i] = 0
  
  for i in items:
    # where the item can be added?
    gr_nums = [g for g in groups.keys() if groups[g] + i <= capacity]
    
    if len(gr_nums) > 0:
      if len(gr_nums) == 1:
        gr_num = gr_nums[0]
        gtype.append( gr_num )
        groups[ gr_num ] += i
      else:
        threshold = 1 / len(gr_nums)
        gr_num = gr_nums[math.floor(rnd_func() / threshold)]
        gtype.append(gr_num)
        groups[gr_num] += i
    
    else:
      return bpGentype4(items, capacity, k+1, rnd_func)
  
  return gtype

def genOk(items, capacity, gtype):
  groups_num = max(gtype) + 1
  for gr_num in range(groups_num):
    group = [i for i,g in zip(items, gtype) if g == gr_num]
    cap = sum([i for i,g in zip(items,gtype) if g == gr_num])
    if cap > capacity:
      return False
  return True

def subst(what, for_what, value):
  if value == for_what:
    return what
  return value

# zmienia numerację grup w genotypie w taki sposób, aby numery
# grup zaczynały się od zera i wzrastały o 1
# np. dla genotypu [1, 2, 2, 1, 3, 2, 3]
# zwróci genotyp [1, 2, 2, 1, 0, 2, 0]
def regroupGtype(gtype):
  max_group = max(gtype)
  for i in range(max_group):
    if i not in gtype:
      gtype = [subst(i, max_group, j) for j in gtype]
      return regroupGtype(gtype)
  return gtype

# krzyżowanie:
# [1,1, | 2,3,3]
# [1,2, | 1,2,3]
def crossOnePointRaw(items, max_capacity, gtype1, gtype2, rnd_func):
  n = len(items)
  point = rnd_int(rnd_func, 1, n-1)
  child = gtype1[:]
  child[point:] = gtype2[point:]
  return regroupGtype(child)

def crossoverOnePoint(items, max_capacity, gtype1, gtype2, rnd_func):
  child1 = crossOnePointRaw(items, max_capacity, gtype1, gtype2, rnd_func)
  child2 = crossOnePointRaw(items, max_capacity, gtype2, gtype1, rnd_func)
  return child1, child2

def mutationOnePoint(items, max_capacity, gtype, rnd_func):
  n = len(items)
  point1 = rnd_int(rnd_func, 0, n-1)
  point2 = rnd_int(rnd_func, 0, n-1)
  new_gtype = gtype[:]
  new_gtype[point1], new_gtype[point2] = gtype[point2], gtype[point1]
  return new_gtype

