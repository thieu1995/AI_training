from models import GA
from utils.GraphUtils import draw_error
import os

low_up = [-10, 10]
epoch = 3000
pop_size = 500
pc = 0.85
pm = 0.025
problem_size = 50
train_loss = []
print_loss = False


ga_para = {
            "epoch": epoch, "pop_size": pop_size, "pc": pc, "pm": pm, "problem_size": problem_size,
            "low_up": low_up, "print_loss": print_loss, "train_loss": train_loss
        }
ga = GA.Wheels(ga_para)
best_chromosome, train_loss, time_per_epoch = ga.train()

draw_error(1, train_loss, "GA_wheels_onePoint_flipPoint", round(best_chromosome[1], 4), time_per_epoch,
           "GA_wheels_onePoint_flipPoint", os.getcwd() + "/week_7-tut_optimization/results/")


