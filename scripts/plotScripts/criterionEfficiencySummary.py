import pickle
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
def plotCriterionEfficiency(tag):
    with open("/t3home/gcelotto/ggHbb/outputs/dict_criterionEfficiency%s.pkl"%tag, 'rb') as file:
        criterionSummary = pickle.load(file)
    oneTimeTrue = True
    fig, ax = plt.subplots(1, 1)
    for key, value in criterionSummary.items():
        non_matched = value[0]
        matched = value[1]
        correct_choice = value[2]
        wrong_choice = value[3]
        noDijetWithTrigMuon = value[4]
        outOfEta = value[5]
        total = matched + non_matched
        assert(matched == correct_choice + wrong_choice + noDijetWithTrigMuon + outOfEta)


        # matched
        ax.bar(x=key, height=correct_choice/total, color='green', label='Correct Choice')
        ax.bar(x=key, height=wrong_choice/total, color='red', label='Wrong Choice', bottom=correct_choice/total)
        ax.bar(x=key, height=noDijetWithTrigMuon/total, color='blue', label='No dijet selected', bottom=(correct_choice+wrong_choice)/total)
        ax.bar(x=key, height=outOfEta/total, color='violet', label='Correct Jets $|\eta|>2.5$ (%.1f%%)'%(outOfEta/(non_matched+matched)*100), bottom=(correct_choice+wrong_choice+noDijetWithTrigMuon)/total)
        # non matched
        ax.bar(x=key, height=non_matched/total, color='black', label='Higgs daughters not matched (%.1f%%)'%(non_matched/(non_matched+matched)*100), bottom=matched/total)
        ax.text(x=key-0.25, y=1.12, s="%.1f%%"%(correct_choice/matched*100))
        ax.text(x=key-0.25, y=1.04, s="%.1f%%"%(correct_choice/(matched+non_matched)*100))
        if oneTimeTrue:
            ax.set_ylim(0, ax.get_ylim()[1]*1.5)
            ax.legend()
            oneTimeTrue=False
        if key==3:
            ax.set_ylim(ax.get_ylim()[0], ax.set_ylim()[1]*1.25)
        ax.set_xlabel("Max Number of Jets", fontsize=24)
        ax.set_ylabel("Fraction of Events", fontsize=24)

    ax.text(x=key+1-0.25, y=1.12, s="Correct / Matched")
    ax.text(x=key+1-0.25, y=1.04, s="Correct / Total")
    outFile = "/t3home/gcelotto/ggHbb/outputs/plots/criterionSummary_%s.png"%tag
    fig.savefig(outFile, bbox_inches='tight')
    print("Criterion efficiency summary saved in ", outFile)



if __name__ == "__main__":
    tag=""
    plotCriterionEfficiency(tag)