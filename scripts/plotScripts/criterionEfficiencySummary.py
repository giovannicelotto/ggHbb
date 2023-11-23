import pickle
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
with open("/t3home/gcelotto/ggHbb/outputs/dict_criterionEfficiency.pkl", 'rb') as file:
    criterionSummary = pickle.load(file)

fig, ax = plt.subplots(1, 1)
for key, value in criterionSummary.items():
    non_matched = value[0]
    matched = value[1]
    correct_choice = value[2]
    wrong_choice = value[3]
    noDijetWithTrigMuon = value[4]
    outOfEta = value[5]
    assert(matched == correct_choice + wrong_choice + noDijetWithTrigMuon + outOfEta)
    

    #ax.bar(x=key, height=matched, color='blue', label='Matched Events')
    ax.bar(x=key, height=non_matched, color='black', label='Higgs daugheters not matched (%.1f%%)'%(non_matched/(non_matched+matched)*100), bottom=matched)
    ax.bar(x=key, height=outOfEta, color='violet', label='Correct Jets $|\eta|>2.5$ (%.1f%%)'%(outOfEta/(non_matched+matched)*100), bottom=correct_choice+wrong_choice+noDijetWithTrigMuon)
    ax.bar(x=key, height=correct_choice, color='green', label='Correct Choice')
    ax.bar(x=key, height=wrong_choice, color='red', label='Wrong Choice', bottom=correct_choice)
    ax.bar(x=key, height=noDijetWithTrigMuon, color='blue', label='No dijet with a Trig muon in the first N Jet', bottom=correct_choice+wrong_choice)
    ax.text(x=key-0.25, y=(non_matched + matched)*1.12, s="%.1f%%"%(correct_choice/matched*100))
    ax.text(x=key-0.25, y=(non_matched + matched)*1.04, s="%.1f%%"%(correct_choice/(matched+non_matched+outOfEta)*100))
    if key==2:
        ax.set_ylim(ax.get_ylim()[0], ax.set_ylim()[1]*1.75)
        ax.legend()
    ax.set_xlabel("Max Number of Jets", fontsize=24)
    ax.set_ylabel("Events", fontsize=24)

fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/criterionSummary.png", bbox_inches='tight')