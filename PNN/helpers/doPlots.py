import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from scaleUnscale import scale
from sklearn.metrics import roc_curve, auc
import pandas as pd
def plot_lossTorch(train_loss_history, val_loss_history, 
              train_classifier_loss_history, val_classifier_loss_history,
              train_dcor_loss_history, val_dcor_loss_history,
              best_epoch,
              outFolder):
    """
    Plots the loss histories for training and validation.

    Parameters:
        train_loss_history (list or np.array): Training loss history.
        val_loss_history (list or np.array): Validation loss history.
        train_classifier_loss_history (list or np.array): Training classifier loss history.
        val_classifier_loss_history (list or np.array): Validation classifier loss history.
        train_dcor_loss_history (list or np.array): Training dCor loss history.
        val_dcor_loss_history (list or np.array): Validation dCor loss history.
        best_epoch (int): Number of the epoch where the best weights were found
        outFolder (str): Output folder path to save the plot.
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(train_loss_history, label='Train Loss')
    ax.plot(val_loss_history, label='Validation Loss', linestyle='dashed')

    ax.plot(train_classifier_loss_history, label='Train Classifier Loss')
    ax.plot(val_classifier_loss_history, label='Validation Classifier Loss', linestyle='dashed')

    ax.plot(np.array(train_dcor_loss_history), label='Train dCor Loss')
    ax.plot(np.array(val_dcor_loss_history), label='Validation dCor Loss', linestyle='dashed')

    ax.set_yscale('log')
    ax.set_ylim(ax.get_ylim())
    ax.set_xlim(ax.get_xlim())
    if best_epoch is not None:
        ax.vlines(x=best_epoch, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dashed', label='Best Epoch')
    ax.legend()

    # Save the figure
    fig.savefig(f"{outFolder}/performance/loss.png")
    plt.close(fig)

def doPlotLoss(fit, outName, earlyStop, patience):

    # "Loss"
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    ax.plot(fit.history['loss'])
    ax.plot(fit.history['val_loss'])
    #ax.plot(fit.history['accuracy'])
    #ax.plot(fit.history['val_accuracy'])
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    
    # plt.yscale('log')
    ax.set_ylim(ymax = max(min(fit.history['loss']), min(fit.history['val_loss']))*1.4, ymin = min(min(fit.history['loss']),min(fit.history['val_loss']))*0.9)
    #ax.set_ylim(ymin= min(min(fit.history['loss']),min(fit.history['val_loss']))*0.9, ymax=max(min(fit.history['accuracy']), min(fit.history['val_accuracy']))*1.4)
    #ax.set_ylim(0, 1)
    ymax = 1#min(fit.history['val_loss'])
    ymin = 0#plt.ylim()[0]
    plt.arrow(x=earlyStop.stopped_epoch-patience+1, y=ymax, dx=0, dy=ymin-ymax, length_includes_head=True, head_length=0.033*(ymin-ymax))
    #plt.legend(['Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy'], loc='upper right')
    plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
    plt.savefig(outName)
    plt.cla()
    print("Saved loss function ", outName)
    print("Saved in ", outName)




def roc(thresholds, signal_predictions, realData_predictions, weights_signal, signalTrain_predictions, realDataTrain_predictions, weights_signalTrain, outName):
    if signalTrain_predictions is not None:
        fpr_train, tpr_train, thresholds = roc_curve(
        y_true=np.concatenate([np.ones(len(signalTrain_predictions)), np.zeros(len(realDataTrain_predictions))]),
        y_score=np.concatenate([signalTrain_predictions, realDataTrain_predictions]),
        sample_weight=np.concatenate([weights_signalTrain, np.ones(len(realDataTrain_predictions))]))
        roc_auc_train = auc(fpr_train, tpr_train)

    fpr, tpr, thresholds = roc_curve(
    y_true=np.concatenate([np.ones(len(signal_predictions)), np.zeros(len(realData_predictions))]),
    y_score=np.concatenate([signal_predictions, realData_predictions]),
    sample_weight=np.concatenate([weights_signal, np.ones(len(realData_predictions))]))
    roc_auc = auc(fpr, tpr)
    
    #tpr, tpr_train = [], []
    #fpr, fpr_train = [], []
    #for t in thresholds:
    #    tpr.append(np.sum(signal_predictions > t)/len(signal_predictions))
    #    fpr.append(np.sum(realData_predictions > t)/len(realData_predictions))
    #    if signalTrain_predictions is not None:
    #        tpr_train.append(np.sum(signalTrain_predictions > t)/len(signalTrain_predictions))
    #        fpr_train.append(np.sum(realDataTrain_predictions > t)/len(realDataTrain_predictions))
    #tpr, fpr =np.array(tpr), np.array(fpr)
    ## auc
    #auc = -simpson(tpr, fpr)
    #if signalTrain_predictions is not None:
    #    tpr_train, fpr_train =np.array(tpr_train), np.array(fpr_train)
    #    auc_train = -simpson(tpr_train, fpr_train)
    #print("AUC : ", auc)


    fig, ax = plt.subplots(1, 1)
    ax.plot(fpr, tpr, marker='o', markersize=1, label='Validation')
    if signalTrain_predictions is not None:
        ax.plot(fpr_train, tpr_train, marker='o', markersize=1, label='Train')
    ax.plot(thresholds, thresholds, linestyle='dotted', color='green')
    
    ax.grid(True)
    ax.set_ylabel("Signal Efficiency")
    ax.set_xlabel("Background Efficiency")
    ax.set_xlim(1e-5,1)
    ax.set_ylim(1e-5,1)
    ax.text(x=0.95, y=0.32, s="AUC Test : %.3f"%roc_auc, ha='right')
    if signalTrain_predictions is not None:
        ax.text(x=0.95, y=0.28, s="AUC Train: %.3f"%roc_auc_train, ha='right')
    ax.legend()
    hep.cms.label()
    if outName is not None:
        fig.savefig(outName, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')

def NNoutputs(signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outName):
    #signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions = np.arctanh(signal_predictions), np.arctanh(realData_predictions), np.arctanh(signalTrain_predictions), np.arctanh(realDataTrain_predictions)
    fig, ax = plt.subplots(1, 1)
    bins=np.linspace(0, 1, 20)
    
    # Hist the predictions
    sig_test_counts = np.histogram(signal_predictions, bins=bins)[0]
    bkg_test_counts = np.histogram(realData_predictions, bins=bins)[0]
    sig_train_counts = np.histogram(signalTrain_predictions, bins=bins)[0]
    bkg_train_counts = np.histogram(realDataTrain_predictions, bins=bins)[0]
    
    # Compute errors as poissonian
    sig_train_counts_err = np.sqrt(sig_train_counts)
    bkg_train_counts_err = np.sqrt(bkg_train_counts)
    sig_test_counts_err = np.sqrt(sig_test_counts)
    bkg_test_counts_err = np.sqrt(bkg_test_counts)
    

    sig_test_counts_err, bkg_test_counts_err = sig_test_counts_err/np.sum(sig_test_counts), bkg_test_counts_err/np.sum(bkg_test_counts)
    sig_train_counts_err, bkg_train_counts_err = sig_train_counts_err/np.sum(sig_train_counts), bkg_train_counts_err/np.sum(bkg_train_counts)
    sig_test_counts, bkg_test_counts, sig_train_counts, bkg_train_counts = sig_test_counts/np.sum(sig_test_counts), bkg_test_counts/np.sum(bkg_test_counts), sig_train_counts/np.sum(sig_train_counts), bkg_train_counts/np.sum(bkg_train_counts)
    
    ax.hist(bins[:-1], bins=bins, weights=sig_test_counts, linewidth=2, histtype='step', label='Signal Validation', color='red')
    ax.hist(bins[:-1], bins=bins, weights=bkg_test_counts, linewidth=2, histtype='step', label='Data Validation', color='blue')
    ax.errorbar(x=(bins[1:]+bins[:-1])/2, y=sig_test_counts, yerr=sig_test_counts_err, linestyle='none', color='red')
    ax.errorbar(x=(bins[1:]+bins[:-1])/2, y=bkg_test_counts, yerr=bkg_test_counts_err, linestyle='none', color='blue')
    
    ax.errorbar(x=(bins[1:]+bins[:-1])/2, y=sig_train_counts, yerr=sig_train_counts_err, linestyle='none', marker='o',label='Signal Train', color='C1')
    ax.errorbar(x=(bins[1:]+bins[:-1])/2, y=bkg_train_counts, yerr=bkg_train_counts_err, linestyle='none', marker='o',label='Data Train', color='C0')
    ax.legend(loc='upper center')
    ax.set_yscale('log')
    ax.set_xlabel("NN output")
    hep.cms.label()
    fig.savefig(outName, bbox_inches='tight')

def getShapTorch(Xtest, model, outName, nFeatures, class_names='NN output', tensor=None):
    from shap import GradientExplainer
    featuresForTraining = Xtest.columns.values
    data = Xtest if tensor is None else tensor
    explainer = GradientExplainer(model=model, data=data)

    shap_values = explainer.shap_values(data)
    # get the contribution for each class by averaging over all the events
    shap_values_average = abs(shap_values).mean(axis=0)

    #find the leading features:
    shap_values_average_sum = shap_values_average.sum(axis=1)

    indices = np.argsort(shap_values_average_sum)[::-1]
    ordered_featuresForTraining = np.array(featuresForTraining)[indices][:nFeatures]
    orderd_shap_values_average = shap_values_average[indices][:nFeatures]

    fig, ax =plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    bins = np.arange(len(ordered_featuresForTraining)+1)
    c0=ax.hist(bins[:-1],bins=bins, width=0.3,color='C0', weights=orderd_shap_values_average ,label=class_names)[0]
    ax.set_xticks(np.arange(len(ordered_featuresForTraining)), labels=ordered_featuresForTraining)
    featureImportance = {}
    for feature, shap in zip(ordered_featuresForTraining, orderd_shap_values_average):
        featureImportance[feature] = shap
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.legend()
    ax.set_ylabel("Mean(|SHAP|)")
    plt.savefig(outName)
    
def getShapNew(Xtest, model, outName, nFeatures, class_names='NN output'):
    from shap import GradientExplainer
    featuresForTraining = Xtest.columns.values
    explainer = GradientExplainer(model=model, data=Xtest)

    shap_values = explainer.shap_values(np.array(Xtest))
    # get the contribution for each class by averaging over all the events
    shap_values_average = abs(shap_values).mean(axis=0)

    #find the leading features:
    shap_values_average_sum = shap_values_average.sum(axis=1)

    indices = np.argsort(shap_values_average_sum)[::-1]
    ordered_featuresForTraining = np.array(featuresForTraining)[indices][:nFeatures]
    orderd_shap_values_average = shap_values_average[indices][:nFeatures]

    fig, ax =plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    bins = np.arange(len(ordered_featuresForTraining)+1)
    c0=ax.hist(bins[:-1],bins=bins, width=0.3,color='C0', weights=orderd_shap_values_average ,label=class_names)[0]
    ax.set_xticks(np.arange(len(ordered_featuresForTraining)), labels=ordered_featuresForTraining)
    featureImportance = {}
    for feature, shap in zip(ordered_featuresForTraining, orderd_shap_values_average):
        featureImportance[feature] = shap
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.legend()
    ax.set_ylabel("Mean(|SHAP|)")
    plt.savefig(outName)


    return featureImportance



def ggHscoreScan(Xtest, Ytest, YPredTest, Wtest, outName):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

    bins = np.linspace(0, 300, 100)
    t = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
    assert len(Ytest)==len(YPredTest)

    Ytest=Ytest.reshape(-1)
    YPredTest=YPredTest.reshape(-1)


    for i in range(len(t)):
            combinedMask = (Ytest==0) & (YPredTest>t[i]) 
            ax.hist(Xtest.dijet_mass[combinedMask], bins=bins, weights=Wtest[combinedMask], label='ggH score >%.1f'%t[i], histtype=u'step', density=True)[0]

    ax.legend()
    ax.set_title("Dijet Mass : ggH score scan")
    if outName is not None:
        fig.savefig(outName, bbox_inches='tight')


def runPlots(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, featuresForTraining, model, inFolder, outFolder):
    signal_predictions = YPredTest[Ytest==1]
    realData_predictions = YPredTest[Ytest==0]
    signalTrain_predictions = YPredTrain[Ytrain==1]
    realDataTrain_predictions = YPredTrain[Ytrain==0]
    
    roc(thresholds=np.linspace(0, 1, 100), signal_predictions=signal_predictions, realData_predictions=realData_predictions, weights_signal=Wtest[Ytest==1],
        signalTrain_predictions=signalTrain_predictions, realDataTrain_predictions=realDataTrain_predictions, weights_signalTrain=Wtrain[Ytrain==1],
        outName=outFolder+"/performance/roc.png")
    NNoutputs(signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outFolder+"/performance/output.png")
    ggHscoreScan(Xtest=Xtest, Ytest=Ytest, YPredTest=YPredTest, Wtest=Wtest, outName=outFolder + "/performance/ggHScoreScan.png")
    # scale
    Xtest  = scale(Xtest, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, featuresForTraining=featuresForTraining)        
    getShapNew(Xtest=Xtest[featuresForTraining].head(1000), model=model, outName=outFolder+'/performance/shap.png', nFeatures=15, class_names=['NN output'])



def runPlotsTorch(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, featuresForTraining, model, inFolder, outFolder, genMassTrain, genMassTest):
    import torch
    # Split predictions for signal and Data
    signal_predictions = YPredTest[Ytest==1]
    realData_predictions = YPredTest[Ytest==0]
    signalTrain_predictions = YPredTrain[Ytrain==1]
    realDataTrain_predictions = YPredTrain[Ytrain==0]

    h125_trainPredictions = YPredTrain[genMassTrain==125]
    h125_testPredictions = YPredTest[genMassTest==125]
    
    roc(thresholds=np.linspace(0, 1, 100), signal_predictions=signal_predictions, realData_predictions=realData_predictions, weights_signal=Wtest[Ytest==1],
        signalTrain_predictions=signalTrain_predictions, realDataTrain_predictions=realDataTrain_predictions, weights_signalTrain=Wtrain[Ytrain==1],
        outName=outFolder+"/performance/roc_ggS0.png")
    roc(thresholds=np.linspace(0, 1, 100), signal_predictions=h125_testPredictions, realData_predictions=realData_predictions, weights_signal=Wtest[genMassTest==125],
        signalTrain_predictions=h125_trainPredictions, realDataTrain_predictions=realDataTrain_predictions, weights_signalTrain=Wtrain[genMassTrain==125],
        outName=outFolder+"/performance/roc_h125.png")
    
    NNoutputs(signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outFolder+"/performance/output_ggS0.png")
    NNoutputs(h125_testPredictions, realData_predictions, h125_trainPredictions, realDataTrain_predictions, outFolder+"/performance/output_125.png")

    ggHscoreScan(Xtest=Xtest, Ytest=Ytest, YPredTest=YPredTest, Wtest=Wtest, outName=outFolder + "/performance/ggHScoreScan.png")
    # scale
    Xtest  = scale(Xtest, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, featuresForTraining=featuresForTraining)
    #X_tensor_test = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()
    #getShapTorch(Xtest=Xtest[featuresForTraining].head(1000), model=model, outName=outFolder+'/performance/shap.png', nFeatures=15, class_names=['NN output'], tensor=X_tensor_test)
