from ctypes import util
import torch
import utilities
import numpy as np
from label_prop_arm import LabelProp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
import sys

"""Methods for testing arms"""

def test_label_prop(datasets, maskings, arm, epochs=100, repeats=3, batch_size=35):
    """method testing label propagation for any amount of datasets and maskings
        datasets: [(X,y),(X,y),...]
        maskings: [.2, .3,...]
    """
    
    all_results = []

    for dataset in datasets:
        X = dataset[0]
        y = dataset[1]

        X = utilities.preprocess(X, scale=False)

        dataset_results = []
        for mask in maskings:
            nah, masked_y = utilities.mask_labels(y, mask)

            train_dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(masked_y))
            test_dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
           
            accuracies = []
            cms = []
            for i in range(repeats):
                arm.reset()
                arm.train(train_dataloader, epochs, verbose=False)
                accuracy, cm = arm.validation_metrics(test_dataloader)

                accuracies.append(accuracy)
                cms.append(cm)

            #get highest accuracy
            best_accuracy = max(accuracies)
            best_cm = cms[accuracies.index(best_accuracy)]

            dataset_results.append((mask, best_accuracy, best_cm))
       
        all_results.append(dataset_results)

    return all_results
            

def make_plot(results, neighbors):
    """makes plot of dataset vs accuracy showing masking differences"""
    results = np.array(results, dtype=object)
    results_df = pd.DataFrame({"dataset":np.repeat(list(range(len(results))), len(results[0])),"masking":results[:,:,0].flatten(), "accuracy":results[:,:,1].flatten()})
    results_df["dataset"] = results_df["dataset"].astype("string")

    plt.scatter(results_df['dataset'], results_df['accuracy'], c=results_df['masking'], cmap=plt.cm.viridis)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy after training 150 epochs")
    clb = plt.colorbar()
    clb.ax.set_title('Masking %')
    plt.title("Label Propagation by Dataset: n=" + str(neighbors))
    plt.grid(axis="y")
    plt.savefig("n="+str(neighbors)+".pdf", format="pdf")

if __name__ == "__main__":
    
    #output_path = sys.argv[1]
    neighbors = int(sys.argv[1])
    datasets = []
    print(neighbors)
    dataset_names = []
    for i in np.arange(.1,.5,.1):
        data_path = "simulations/splat_" + str(round(i,5)) + "_de/counts.csv"
        meta_path = "simulations/splat_" + str(round(i,5)) + "_de/meta.csv"
        temp_data = pd.read_csv(data_path, index_col=0)
        temp_meta = pd.read_csv(meta_path, index_col=0)
        
        X = np.array(temp_data)
        y = pd.factorize(temp_meta['Group'], sort=True)[0]
        datasets.append((X,y))
        dataset_names.append(str(round(i,5)))

    maskings = [.6,.7, .8, .9, .95, .99]
    arm = LabelProp("configs/semi_basic_linear.txt", neighbors)
    results = test_label_prop(datasets, maskings, arm, epochs=150)
    make_plot(results, neighbors)
    """
    x = 0
    for dataset_results in results:
        dir_path = output_path + "_dataset_" + dataset_names[x]
        print("Dataset " + dataset_names[x])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        dataset_results = np.array(dataset_results, dtype=object)
        
        plt.scatter(dataset_results[:,0], dataset_results[:,1])
        plt.xlabel("Masking Percentage")
        plt.ylabel("Best Accuracy over 3 training runs")
        plt.title("Masking percent vs Accuracy: Trained 100")
        plt.savefig(dir_path + "/masking_plot.pdf")
        
        for masking in dataset_results:
            print("Masking: " + str(masking[0]))
            print("Best Accuracy: " + str(masking[1]))
            print(masking[2])
            print()
            #disp = ConfusionMatrixDisplay(confusion_matrix=masking[2])
            #print(type(disp))
            #print(type(disp.plot()))
            #plt.show()
            #plt.savefig((dir_path + "/confusion_matrix_" + str(masking[0])))
        
        x +=1
    """
