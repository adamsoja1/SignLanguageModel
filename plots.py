
import matplotlib.pyplot as plt
import seaborn as sns

def make_plot(history,metric,name):
    """
    Accuracy plot of model 
    """

    sns.set()
    acc, val_acc = history[metric],history[f'val_{metric}']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(20, 13))
    plt.plot(epochs, acc, label=f'Training {name}', marker='o')
    plt.plot(epochs, val_acc, label=f'Validation {name}', marker='o')
    plt.legend()
    plt.title('Training and validation')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric}')
    plt.savefig(f'{name}.png')
