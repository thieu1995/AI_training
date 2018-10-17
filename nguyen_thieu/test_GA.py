from models import GA

low_up = [-5, 5]
epoch = 2000
pop_size = 1000
pc = 0.85
pm = 0.5
problem_size = 10
train_loss = []
print_loss = True


ga_para = {
            "epoch": epoch, "pop_size": pop_size, "pc": pc, "pm": pm, "problem_size": problem_size,
            "low_up": low_up, "print_loss": print_loss, "train_loss": train_loss
        }
ga = GA.BaseClass(ga_para)
chromosome, loss_train = ga.train()