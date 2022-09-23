# task-scheduling
Processor task scheduling using AI algorithms and analyzing the results

![output screenshot](https://github.com/pard1s/task-scheduling/blob/main/screenshot.jpg)

In this program, you must first enter the number of tasks and nodes. Each task will then be assigned a random deadline (in seconds) and a cost. Five AI algorithms have been implemented to assign all tasks to noeds in such a way that we can complete as many tasks as possible before their deadlines at the lowest possible cost.
The algorithms are:
  - Greedy
  - Hill Climbing
  - Random Restart Hill climbing
  - simulated anealing
  - Genetic
  
By choosing one of the above options, you will see:
  - the effincy of that algorithm ( based on number of tasks that were finished before their deadline)
  - number of tasks that finished before their deadline
  - number of tasks that finished after their deadline
  - and the total cost
  
Then, by pressing '1', an excel file with more details and a chart demonstrating the efficiency of the algorithms will be generated.

Please install required packages before running:
    - pip install xlsxwriter
    - pip install colorama
    - pip install pandas
