import numpy as np
import matplotlib.pyplot as plt

#  Координаты городов

'''coordinates = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
                        [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0]])'''

coordinates = np.array([[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],
                [880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],
                [1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,  5.0],[845.0,680.0],
                [725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],
                [300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],
                [1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0]])

'''coordinates = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
                        [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
                        [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
                        [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
                        [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
                        [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
                        [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
                        [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
                        [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
                        [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0]])'''

def getdistmat(coordinates):
    num = coordinates.shape[0]
    #distmat = np.zeros((10,10))
    distmat = np.zeros((30, 30))
    #distmat = np.zeros((50, 50))
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] = distmat[j][i] = np.linalg.norm(
                coordinates[i] - coordinates[j])
    return distmat

# Инициализация
distmat = getdistmat(coordinates)
numant = 10  # количество муравьёв
numcity = coordinates.shape[0] # Количество городов
alpha = 1 #  Фактор важности феромона
beta = 5 #  Фактор важности эвристической функции
rho = 0.1 #  Скорость испарения феромонов
Q = 1 # Общее высвобождение феромонов
iter = 0 # Счётчик итераций
itermax = 100 # Максимальное число итераций
etatable = 1.0 / (distmat + np.diag([1e10] * numcity)) # Эвристическая функция
pathtable = np.zeros((numant, numcity)).astype(int) # Запись путей
pheromonetable = np.ones((numcity, numcity)) # Матрица феромонов
distmat = getdistmat(coordinates) # Матрица расстояний
lengthaver = np.zeros(itermax) # Длины путей в каждой итерации
lengthbest = np.zeros(itermax) # Наименьшая длина пути в каждой итерации
pathbest = np.zeros((itermax, numcity)) # Кратчайший путь в каждой итерации

# НАЧАЛО ЦИКЛА ОБРАБОТКИ КООРДИНАТ
while iter < itermax:
    # Выбираем случайный стартовый город для каждого муравья
    if numant <= numcity:
        # Городов больше, чем муравьёв
        pathtable[:, 0] = np.random.permutation(range(0, numcity))[:numant]
    else:
        # Муравьёв больше, чем городов
        pathtable[:numcity, 0] = np.random.permutation(range(0, numcity))[:]
        pathtable[numcity:, 0] = np.random.permutation(range(0, numcity))[:numant - numcity]
    length = np.zeros(numant)  # Вычисление пройденного пути
    for i in range(numant):
        visiting = pathtable[i, 0]  # Текущий город
        unvisited = set(range(numcity))  # Непосещённые города
        unvisited.remove(visiting)  # Обновление списка непосещённых городов
        for j in range(1, numcity):
            # Выбор следующего города для посещения
            listunvisited = list(unvisited) # Список непосещённых городов
            probtrans = np.zeros(len(listunvisited))
            for k in range(len(listunvisited)):
                probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]], alpha) \
                    * np.power(etatable[visiting][listunvisited[k]], beta) # Расчёт следующего города
            cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()
            cumsumprobtrans -= np.random.rand()
            k = listunvisited[(np.where(cumsumprobtrans > 0)[0])[0]]
            pathtable[i, j] = k # Добавить путь в таблицу путей
            unvisited.remove(k) # Удалить город из списка непосещённых
            length[i] += distmat[visiting][k] # Длина пути
            visiting = k
        # Длина пройденного пути
        length[i] += distmat[visiting][pathtable[i, 0]]
        # Конец итерации, вычисление длин
    lengthaver[iter] = length.mean()
    if iter == 0:
        lengthbest[iter] = length.min()
        pathbest[iter] = pathtable[length.argmin()].copy()
    else:
        if length.min() > lengthbest[iter - 1]:
            lengthbest[iter] = lengthbest[iter - 1]
            pathbest[iter] = pathbest[iter - 1].copy()
        else:
            lengthbest[iter] = length.min()
            pathbest[iter] = pathtable[length.argmin()].copy()
    changepheromonetable = np.zeros((numcity, numcity))
    for i in range(numant):
        for j in range(numcity - 1):
            changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / distmat[pathtable[i, j]][
                pathtable[i, j + 1]]  # Перерасчёт феромонов
        changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += Q / distmat[pathtable[i, j + 1]][pathtable[i, 0]]
    pheromonetable = (1 - rho) * pheromonetable + changepheromonetable
    if iter%10==0:
        print("iter( The number of iterations ):", iter)
    iter += 1

# График длины пути
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
axes[0].plot(lengthaver, 'k', marker=u'')
axes[0].set_title('Средняя длина')
#axes[0].set_xlabel(u'Итерации')

# График поиска кратчайшего пути
axes[1].plot(lengthbest, 'k', marker=u'')
axes[1].set_title('Кратчайший путь')
axes[1].set_xlabel(u'Итерации')
fig.savefig('average_best.png', dpi=500, bbox_inches='tight')
plt.show()

# Карта кратчайшего пути
bestpath = pathbest[-1]
plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker=u'$\cdot$')
plt.xlim([-100, 2000])
plt.ylim([-100, 1500])

for i in range(numcity - 1):
    m = int(bestpath[i])
    n = int(bestpath[i + 1])
    plt.plot([coordinates[m][0], coordinates[n][0]], [
             coordinates[m][1], coordinates[n][1]], 'k')
plt.plot([coordinates[int(bestpath[0])][0], coordinates[int(n)][0]],
         [coordinates[int(bestpath[0])][1], coordinates[int(n)][1]], 'b')
ax = plt.gca()
ax.set_title("Кратчайший путь")
ax.set_xlabel('Ось X')
ax.set_ylabel('Ось Y')

plt.savefig('best path.png', dpi=500, bbox_inches='tight')
plt.show()