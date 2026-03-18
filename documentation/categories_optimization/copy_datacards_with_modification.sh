for i in {2..8}; do
  cp /t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm/datacard_ttbar_CR_1.txt \
     /t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm/datacard_ttbar_CR_${i}.txt

  sed -i "s/histograms_0/histograms_${i}/g" \
     /t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm/datacard_ttbar_CR_${i}.txt
done