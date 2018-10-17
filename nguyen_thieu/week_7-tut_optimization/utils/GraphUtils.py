
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 14,
        }

def draw_predict(fig_id=None, y_test=None, y_pred=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.ylabel('CPU')
    plt.xlabel('Timestamp')
    plt.legend(['Actual', 'Predict'], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None

def draw_predict_with_error(fig_id=None, y_test=None, y_pred=None, RMSE=None, MAE=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.ylabel('Real value')
    plt.xlabel('Point')
    plt.legend(['Predict y... RMSE= ' + str(RMSE), 'Test y... MAE= ' + str(MAE)], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None

def draw_error(fig_id=None, errors=None, model_name=None, fitness=None, time_per_epoch=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(errors)
    plt.ylabel('Fitness')
    plt.xlabel('Epoch')
    plt.text(0.4, 0.8, 'Best fitness = ' + str(fitness), fontdict=font, transform=plt.gcf().transFigure)
    plt.text(0.4, 0.7, 'Time/epoch = ' + str(time_per_epoch) + ' second', fontdict=font, transform=plt.gcf().transFigure)
    #plt.text(4, 1, 'Best fitness = ' + str(fitness), fontdict=font, ha='left', rotation=15, wrap=True)
    #plt.text(5, 10, 'Time/epoch = ' + str(time_per_epoch) + ' second', fontdict=font, ha='left', rotation=15, wrap=True)
    plt.title(model_name)
    #plt.subplots_adjust(left=0.01)
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None