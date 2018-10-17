
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

def draw_error(fig_id=None, errors=None, model_name=None, fitness=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(errors)
    plt.ylabel('Fitness')
    plt.xlabel('Epoch')
    plt.legend(['Best fitness = ' + str(fitness)], loc='upper right')
    plt.title(model_name)
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None