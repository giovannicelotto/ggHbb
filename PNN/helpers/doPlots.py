import matplotlib.pyplot as plt

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