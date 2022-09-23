# pardis ghavami  - 9717023147  - HW2 - question4

# you have to install the imported library/modules first in order to run the code:
# 1. open cmd -> 2. write "python" -> 3. write "pip install (library name)"
import random
import timeit
import math
import copy
import inspect
import pandas as pd 
from colorama import init
from colorama import Fore, Style
init()
count = 0
tasks = []
nodes = []
m = 0
n = 0
def main():
    global count, tasks, nodes, n, m
    if count == 0:
        count += 1
        print( Fore.MAGENTA + "\n_______________ Task Scheduling _______________\n" , Style.RESET_ALL)
        n = int(input("please enter number of TASKS:\n"))
        m = int(input("please enter number of NODES:\n"))
        tasks = set_tasks(n)
        nodes = set_nodes(m)
        menu(tasks, nodes, n, m)
    else:
        menu(tasks, nodes, n, m)
def set_tasks(n):              #returns 3 parallel lists that each one represents a property of tasks
    instructions = []     #number of instructions
    deadline = []     #task deadline
    finished_time = []
    cost = []
    assigned_node = []
    runtime = []
    for task in range(n):
        instructions.append(random.randrange(1000, 10000))
        deadline.append(random.randrange(500, 10000))
        finished_time.append(0)
        cost.append(0)
        assigned_node.append(-1)
        runtime.append(0)
    return [instructions, deadline, assigned_node, runtime, finished_time, cost]

def set_nodes(m):               #returns 2 parallel lists that each one represents a property of nodes
    speed = []  #process speed
    price = []  #process cost
    waiting_time = []    #waiting time
    tasks = []
    for node in range(m):
        speed.append(random.randrange(1000, 10000))
        price.append(random.uniform(0.1, 1))
        waiting_time.append(0)
        tasks.append(list())
    return [speed, price, waiting_time , tasks]


def menu(tasks, nodes, n, m):
    temp_tasks = copy.deepcopy(tasks)
    temp_nodes = copy.deepcopy(nodes)
    print(Fore.GREEN + "_______________________________________________\n" + Style.RESET_ALL)
    print(Fore.GREEN + "              Available Algorithms             " , Style.RESET_ALL)
    print(Fore.GREEN + "_______________________________________________" , Style.RESET_ALL)
    print(Fore.CYAN + "\n1. Greedy\n2. Hill Climbing\n3. Random Restart Hill climbing\n4. Simulated anealing\n5. Genetic\n6. EXIT" , Style.RESET_ALL)
    print(Fore.GREEN +               "_______________________________________________" , Style.RESET_ALL)
    choice = int(input("\n\nEnter your selection:\n"))
    switcher = {
        1 : greedy,
        2 : hill_climbing,
        3 : rand_restart_hill_climbing,
        4 : simulated_anealing,
        5 : genetic,
        6 : Exit
    }
    choice = switcher.get(choice , no_such_alg)
    choice(temp_tasks, temp_nodes, n, m)

def show_details(tasks, nodes, n, m):
    print('{:^8}'.format('task'), '{:^15}'.format('instructions'), '{:^15}'.format('deadline'), '{:^15}'.format('assigned to node'), '{:^15}'.format('run time'), '{:^15}'.format('finished time'), '{:^15}'.format('cost'))
    for i in range(n):
        print('{:^8}'.format(i), '{:^15}'.format(tasks[0][i]), '{:^15}'.format(tasks[1][i]), '{:^15}'.format(tasks[2][i]), '{:^15}'.format('{:.2f}'.format(tasks[3][i])), '{:^15}'.format('{:.2f}'.format(tasks[4][i])), '{:^15}'.format('{:.2f}'.format(tasks[5][i])))

    print("\n___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n")
    print('{:^8}'.format('node'), '{:^15}'.format('speed'), '{:^15}'.format('price'), '{:^15}'.format('waiting time'), '{:^15}'.format('tasks'))
    for i in range(m):
        print('{:^8}'.format(i), '{:^15}'.format(nodes[0][i]), '{:^15}'.format('{:.2f}'.format(nodes[1][i])), '{:^15}'.format('{:.2f}'.format(nodes[2][i])), nodes[3][i])
    plot_charts(tasks, n, inspect.stack()[1][3])
    print(Fore.RED, "\n\nThe .xlsx file created in the relative path, please close the .xlsx window before trying another algotithm!", Style.RESET_ALL)
    input("press a key to continue.\n")

#############################################################################################################################################

def greedy(tasks, nodes, n, m):
    tasks[0], tasks[1] = zip(*sorted(zip(tasks[0], tasks[1]), key= lambda item: item[1]))  # sorted by order of task deadline
    nodes[0] ,nodes[1] = zip(*sorted(zip(nodes[0], nodes[1]), key = lambda item: item[1]))  # sorted by order of node price
    count = 0  #number of tasks that won't be finished before their deadline
    queue = []
    total_cost = 0
    max = int( n / m ) + 1
    for i in range(n):
        for j in range(m):
            tasks[3][i] = tasks[0][i] * 1000 / nodes[0][j]   #task runtime
            if (tasks[3][i] + nodes[2][j]) <= tasks[1][i]:   # if task.runTime + node.waitingTime <= task.deadline THEN:
                update_information(tasks, nodes, i, j)       #task i assigned to node j
                total_cost += tasks[5][i]                           # task's cost added to the total cost
                break
            elif j == (m - 1) and tasks[2][i] == -1:
                queue.append(i)   #adds tasks that can't be finished before their deadline to the queue, they will be assigned to cheapest nodes after asigning other tasks to nodes
    count = len(queue)
    while queue:                  # tasks that could not be finished before their deadline will be distributed on nodes
        i = queue.pop(0)
        j = 0
        while(j < m):
            if len(nodes[3][j]) < max:        # to avoid assigning all of the tasks to only the cheapest node, we consider a limit for the nodes
                update_information(tasks, nodes, i , j)  #task i assigned to node j
                total_cost += tasks[5][i]
                break
            j += 1
    print("efficiency: ", (n - count) / n * 100)
    print("number of tasks that finished before their deadline: ", n - count, "\nand number of tasks that finished after their deadline: ", count)
    print("total cost : ", '{:.2f}'.format(total_cost))
    if input("if you want to see more details press 1 other wise press any key\n") == '1':
        show_details(tasks, nodes, n, m)
    main()

###############################################################################################################################################

def hill_climbing(tasks, nodes, n, m):
    # generate inital state randomly
    for i in range(n):
        update_information(tasks, nodes, i, random.randrange(0, m))
   #################################### Climbing! ####################################
    count = 0  #generated neighbors
    value = objective_funct(tasks)
    changed_nodes = []
    total_count = 0
    for i in range(n):
        changed_nodes.append(list())
    while value[0] != 0:
        if count >= 80 :
            break
        t = random.randrange(0, n)  # a random task index to change
        z = random.randrange(0,m)   # a random node index to change for a task
        f = 0
        while z in changed_nodes[t]:
            if f == n:
                break
            if len(changed_nodes[t]) == m:
                t = random.randrange(0, n) 
                f += 1 
            z = random.randrange(0, m) 
        if f == n:
            break
        changed_nodes[t].append(z)
        node = tasks[2][t]
        node_history = del_task_from_node(tasks, nodes, t, node)   #free t's runtime from the node's waiting time
        update_information(tasks, nodes, t, z)
        temp_value = objective_funct(tasks)
        count += 1
        total_count += 1
        if temp_value[0] < value[0] or (temp_value[0] == value[0] and temp_value[1] < value[1]):
            value = temp_value
            for i in changed_nodes:
                i = list()
            count = 0
        else:
            del_task_from_node(tasks,nodes, t,z)
            restore_node(tasks, nodes, node , node_history)

    if( inspect.stack()[1][3] == 'menu'):
        print("efficiency: ", ((n - value[0])/n)*100)
        print("number of tasks that finished before their deadline: ", n - value[0], "\nand number of tasks that finished after their deadline: ", value[0])
        print("total cost : ", '{:.2f}'.format(value[1]))
        if input("if you want to see more details press 1 other wise press any key\n") == '1':
            show_details(tasks, nodes, n, m)
        main()
    if(inspect.stack()[1][3] == 'rand_restart_hill_climbing'):
        return [tasks, nodes, value]

def update_information(tasks, nodes, i, node):
    tasks[2][i]= node           # node saved in task.assigned_node
    nodes[3][node].append(i)    # task i appended in node.tasks
    tasks[3][i] =  tasks[0][i] * 1000 / nodes[0][node]     # task runtime
    tasks[4][i] =  nodes[2][node] + tasks[3][i]   # task finished time = task.runtime + node.waitingTime
    nodes[2][node] += tasks[3][i]   #node waiting time updated  
    tasks[5][i] = (tasks[3][i] / 1000) * nodes[1][node] # task's prcie calculated 

def del_task_from_node(tasks, nodes, t, n):
    t_index = nodes[3][n].index(t)
    i = t_index
    queue = []
    while i <= len(nodes[3][n]) - 1:
        queue.append(nodes[3][n][i])
        i += 1
    i = t_index
    if(t_index == 0):
        nodes[2][n] = 0
    else:
        nodes[2][n] = tasks[4][ nodes[3][n][t_index - 1] ]   # node waiting time = previous task's finsih time
    nodes[3][n].pop(t_index)
    while t_index <= len(nodes[3][n]) - 1:
        t = nodes[3][n][t_index]
        tasks[4][t] =  nodes[2][n] + tasks[3][t]   #finished time
        nodes[2][n] += tasks[3][t]   #node waiting time added
        t_index += 1
    
    return[i, queue]

def restore_node(tasks, nodes, n, node_history):
    i = node_history[0] 
    queue = node_history[1]
    while i <= len(nodes[3][n]) - 1:
        t = nodes[3][n].pop(i)
        nodes[2][n] -= tasks[3][t]

    while queue:
        update_information(tasks, nodes, queue.pop(0), n)


def objective_funct(tasks):
    count = 0   # number of tasks that finished after their deadline time
    total_cost = 0
    for i in range(len(tasks[0])):
        if tasks[4][i] > tasks[1][i]:
            count += 1
        total_cost = total_cost + tasks[5][i]
    return [count, total_cost]
   

###############################################################################################################################################

def rand_restart_hill_climbing(tasks, nodes, n, m):
    restart_times = int(input("\nPlease enter the desired number of restart times:\n"))
    resaults = []
    for i in range(restart_times):
        temp_tasks = copy.deepcopy(tasks)
        temp_nodes = copy.deepcopy(nodes)
        resaults.append(hill_climbing(temp_tasks, temp_nodes, n, m))
    min = 0
    for j in range(len(resaults)):
        if resaults[j][2][0] < resaults[min][2][0] or (resaults[j][2][0] == resaults[min][2][0] and resaults[j][2][1] < resaults[min][2][1] ):
            min = j
    print("efficiency: ", ((n - resaults[min][2][0])/n)*100)
    print("number of tasks that finished before their deadline: ", n - resaults[min][2][0], "\nand number of tasks that finished after their deadline: ", resaults[min][2][0])
    print("total cost : ", '{:.2f}'.format(resaults[min][2][1]))
    if input("if you want to see more details press 1 other wise press any key\n") == '1':
        show_details(resaults[min][0], resaults[min][1], n, m)
    main()
   

###############################################################################################################################################

def simulated_anealing(tasks, nodes, n, m):
    T = int(input("\nPlease enter the initial temprature:\n"))
    final_temp = 0.1 
    alpha = 0.1
    delta_v0 = 0     # Difference in the number of tasks completed after their deadline of two states
    delta_v1 = 0     # Difference in the cost of two states
    current_temp = T
    for i in range(n):
        node = random.randrange(0, m)
        update_information(tasks, nodes, i, node)
    value = objective_funct(tasks)
    ###################################### anealing ####################################
    while current_temp > final_temp:
        t = random.randrange(0, n)  # a random task index to change
        node = tasks[2][t]
        node_history = del_task_from_node(tasks, nodes, t, node)   #free t's runtime from the node's waiting time
        z = random.randrange(0, m)  # a random node index to change for a task
        update_information(tasks, nodes, t, z)
        temp_value = objective_funct(tasks)
        delta_v0 = value[0] - temp_value[0]
        delta_v1 = value[1] - temp_value[1]
        if delta_v0 > 0 or (delta_v0 == 0 and delta_v1 > 0):
            value = temp_value
        else:
            if random.uniform(0, 1) < math.exp(delta_v0 / current_temp):
                value = temp_value
            else: 
                del_task_from_node(tasks,nodes, t,z)
                restore_node(tasks, nodes, node , node_history)
        current_temp -= alpha
    print("efficiency: ", ((n - value[0])/n)*100)
    print("number of tasks that finished before their deadline: ", n - value[0], "\nand number of tasks that finished after their deadline: ", value[0])
    print("total cost : ", '{:.2f}'.format(value[1]))
    if input("if you want to see more details press 1 other wise press any key\n") == '1':
        show_details(tasks, nodes, n, m)
    main()

###############################################################################################################################################

def genetic(tasks, nodes, n, m):
    size = 100
    iterate = int(input("please enter the number of itrations:\n")) 
    # creating a matrix of genes (initial population) :
    population = list()
    for i in range(size):
        gene = []
        for j in range(n):
            gene.append(random.randrange(0, m))             #gene[i] created.
        population.append(gene[:])

    for i in range(iterate):
        weights = fitness_func(population, tasks, nodes, n, m)
        population2 = list()
        for j in range(size):
            parent1, parent2 = weighted_random_choices(population, weights)
            child = reproduce(parent1, parent2)
            mutate(child, m)
            population2.append(child)
        population = population2

    final_weights = fitness_func(population, tasks, nodes, n, m)
    max_value = max(final_weights)
    max_index = final_weights.index(max_value)    # the index of best gene in final population
    for j in range(m):                            # choosing the final best gene as the final resault
        nodes[3][j] = []
        nodes[2][j] = 0
    for j in range(n):
        update_information(tasks, nodes, j, population[max_index][j])
    value = objective_funct(tasks)
    print("efficiency: ", ((n - value[0])/n)*100)
    print("number of tasks that finished before their deadline: ", n - value[0], "\nand number of tasks that finished after their deadline: ", value[0])
    print("total cost : ", '{:.2f}'.format(value[1]))
    if input("if you want to see more details press 1 other wise press any key\n") == '1':
        show_details(tasks, nodes, n, m)
    main()

def fitness_func(population, tasks, nodes, n, m):
    weights = list()  #the fitness value of each gene
    for gene in population:
        for j in range(m):
            nodes[3][j] = []
            nodes[2][j] = 0
        for j in range(n):
            update_information(tasks, nodes, j, gene[j])
        weights.append(n - objective_funct(tasks)[0])       #gene[i] weighted.
    return weights

def weighted_random_choices(matrix, w):
    l = []
    for i in range(len(w)):
        l.append(i)
    choices = random.choices(l, weights = tuple(w), k = 2)
    while choices[0] == choices[1]:
        choices[1] = random.choices(l, weights = tuple(w), k = 1)[0]
    return matrix[choices[0]], matrix[choices[1]]

def reproduce(parent1, parent2):
    n = len(parent1)
    c = random.randrange(1, n)
    child = parent1[:c] + parent2[c:]
    return child[:]

def mutate(child, m):
    chance = 100
    for i in range(len(child)):
        if int(random.random() * chance) == 1:
            new_node = random.randrange(0, m)
            while new_node == child[i]:
                new_node = random.randrange(0, m)
            child[i] = new_node

###############################################################################################################################################

# plot a line chart using pandas
def plot_charts(tasks, n, alg_name):
    dataframe = pd.DataFrame({ 
                    'task deadline' : tasks[1],
                    'task finish time' : tasks[4]
					})  
    writer_object = pd.ExcelWriter('pandas_line_chart.xlsx', engine ='xlsxwriter') 
    dataframe.to_excel(writer_object, sheet_name ='Sheet1') 
    workbook_object = writer_object.book 
    worksheet_object = writer_object.sheets['Sheet1'] 
    worksheet_object.set_column('B:G', 25)
    chart_object = workbook_object.add_chart({'type': 'line'}) 
    chart_object.add_series({ 
        'name':	 ['Sheet1', 0, 1], 
        'categories': ['Sheet1', 1, 0, n+1, 0], 
        'values':	 ['Sheet1', 1, 1, n+1, 1], 
        'line' :   {'color' : 'pink'}
        }) 
    chart_object.add_series({ 
        'name':	 ['Sheet1', 0, 2],
        'categories': ['Sheet1', 1, 0, n+1, 0],  
        'values':	 ['Sheet1', 1, 2, n+1, 2],
        'line' :   {'color' : 'blue'} 
        }) 

    chart_object.set_title({'name': alg_name}) 

    chart_object.set_x_axis({
        'name': 'tasks', 
        
        })        

    chart_object.set_y_axis({
        'name': 'time',
        'major_unit': 500
        }) 

    worksheet_object.insert_chart('D2', chart_object, 
                    {'x_offset': 50, 'y_offset': 50,
                    'x_scale': 5 , 'y_scale': 4
                    }
                    ) 

    writer_object.save() 


#################################################################################################################################################

def no_such_alg(task, nodes, n, m):
    print(Fore.RED + "INVALID INPUT >> PLEASE ENTER AN INTEGER BETWEEN 1 - 6\n", Style.RESET_ALL)
    main()

def Exit(tasks, nodes, n, m):
    return
###############################################################################################################################################
if __name__ == "__main__":
    main()