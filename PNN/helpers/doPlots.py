import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from matplotlib import colors
import mplhep as hep
hep.style.use("CMS")
import shap
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from scaleUnscale import scale

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




def roc(thresholds, signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outName):

    tpr, tpr_train = [], []
    fpr, fpr_train = [], []
    for t in thresholds:
        tpr.append(np.sum(signal_predictions > t)/len(signal_predictions))
        fpr.append(np.sum(realData_predictions > t)/len(realData_predictions))
        tpr_train.append(np.sum(signalTrain_predictions > t)/len(signalTrain_predictions))
        fpr_train.append(np.sum(realDataTrain_predictions > t)/len(realDataTrain_predictions))
    tpr, fpr =np.array(tpr), np.array(fpr)
    tpr_train, fpr_train =np.array(tpr_train), np.array(fpr_train)
    # auc
    auc = -simpson(tpr, fpr)
    auc_train = -simpson(tpr_train, fpr_train)
    print("AUC : ", auc)


    fig, ax = plt.subplots(1, 1)
    ax.plot(fpr, tpr, marker='o', markersize=1, label='test')
    ax.plot(fpr_train, tpr_train, marker='o', markersize=1, label='train')
    ax.plot(thresholds, thresholds, linestyle='dotted', color='green')
    
    ax.grid(True)
    ax.set_ylabel("Signal Efficiency")
    ax.set_xlabel("Background Efficiency")
    ax.set_xlim(1e-5,1)
    ax.set_ylim(1e-1,1)
    ax.text(x=0.95, y=0.32, s="AUC Test : %.3f"%auc, ha='right')
    ax.text(x=0.95, y=0.28, s="AUC Train: %.3f"%auc_train, ha='right')
    ax.legend()
    fig.savefig(outName, bbox_inches='tight')
    hep.cms.label()
    plt.close('all')

def NNoutputs(signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outName):
    #signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions = np.arctanh(signal_predictions), np.arctanh(realData_predictions), np.arctanh(signalTrain_predictions), np.arctanh(realDataTrain_predictions)
    fig, ax = plt.subplots(1, 1)
    bins=np.linspace(0, 1, 20)
    sig_test_counts = np.histogram(signal_predictions, bins=bins)[0]
    bkg_test_counts = np.histogram(realData_predictions, bins=bins)[0]
    sig_train_counts = np.histogram(signalTrain_predictions, bins=bins)[0]
    bkg_train_counts = np.histogram(realDataTrain_predictions, bins=bins)[0]
    sig_train_counts_err = np.sqrt(sig_train_counts)
    bkg_train_counts_err = np.sqrt(bkg_train_counts)
    sig_test_counts, bkg_test_counts, sig_train_counts, bkg_train_counts, sig_train_counts_err, bkg_train_counts_err = sig_test_counts/np.sum(sig_test_counts), bkg_test_counts/np.sum(bkg_test_counts), sig_train_counts/np.sum(sig_train_counts), bkg_train_counts/np.sum(bkg_train_counts), sig_train_counts_err/np.sum(sig_train_counts), bkg_train_counts_err/np.sum(bkg_train_counts)
    ax.hist(bins[:-1], bins=bins, weights=sig_test_counts, alpha=0.3, label='signal test')
    ax.hist(bins[:-1], bins=bins, weights=bkg_test_counts, alpha=.3, label='bkg test')
    ax.errorbar(x=(bins[1:]+bins[:-1])/2, y=sig_train_counts, yerr=sig_train_counts_err, linestyle='none', label='signal train')
    ax.errorbar(x=(bins[1:]+bins[:-1])/2, y=bkg_train_counts, yerr=bkg_train_counts_err, linestyle='none', label='bkg train')
    ax.legend(loc='upper center')
    ax.set_yscale('log')
    ax.set_xlabel("NN output")
    hep.cms.label()
    fig.savefig(outName, bbox_inches='tight')

def getShapNew(Xtest, model, outName, nFeatures, class_names=['NN output']):
    print(Xtest.shape)
    featuresForTraining = Xtest.columns.values
    explainer = shap.GradientExplainer(model=model, data=Xtest)

    shap_values = explainer.shap_values(np.array(Xtest))
    print(shap_values)
    # get the contribution for each class by averaging over all the events
    shap_values_average = abs(shap_values).mean(axis=0)

    #find the leading features:
    shap_values_average_sum = shap_values_average.sum(axis=1)
    for idx, feature in enumerate(featuresForTraining):
        print(feature, shap_values_average_sum[idx])
    indices = np.argsort(shap_values_average_sum)[::-1]
    ordered_featuresForTraining = np.array(featuresForTraining)[indices][:nFeatures]
    orderd_shap_values_average = shap_values_average[indices][:nFeatures]

    fig, ax =plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    bins = np.arange(len(ordered_featuresForTraining)+1)
    c0=ax.hist(bins[:-1],bins=bins, width=0.3,color='C0', weights=orderd_shap_values_average ,label='Data')[0]
    ax.set_xticks(np.arange(len(ordered_featuresForTraining)), labels=ordered_featuresForTraining)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.legend()
    ax.set_ylabel("Mean(|SHAP|)")
    plt.savefig(outName)



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
    #print(Wtest[combinedMask][:10], Wtest[combinedMask].mean(), Wtest[combinedMask].std())
    ax.set_title("Dijet Mass : ggH score scan")
    fig.savefig(outName, bbox_inches='tight')


def runPlots(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, featuresForTraining, model, inFolder, outFolder):
    signal_predictions = YPredTest[Ytest==1]
    realData_predictions = YPredTest[Ytest==0]
    signalTrain_predictions = YPredTrain[Ytrain==1]
    realDataTrain_predictions = YPredTrain[Ytrain==0]

    roc(thresholds=np.linspace(0, 1, 100), signal_predictions=signal_predictions, realData_predictions=realData_predictions, signalTrain_predictions=signalTrain_predictions, realDataTrain_predictions=realDataTrain_predictions, outName=outFolder+"/performance/roc.png")
    NNoutputs(signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outFolder+"/performance/output.png")
    ggHscoreScan(Xtest=Xtest, Ytest=Ytest, YPredTest=YPredTest, Wtest=Wtest, outName=outFolder + "/performance/ggHScoreScan.png")
    # scale
    Xtest  = scale(Xtest, scalerName= inFolder + "/myScaler.pkl" ,fit=False)
    getShapNew(Xtest=Xtest[featuresForTraining].head(1000), model=model, outName=outFolder+'/performance/shap.png', nFeatures=15, class_names=['NN output'])
