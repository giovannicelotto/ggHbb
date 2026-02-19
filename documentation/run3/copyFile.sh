#!/bin/bash
dasgoclient -query="file dataset=/ParkingSingleMuon0/Run2025E-PromptReco-v1/NANOAOD"  > filelist.txt

cat filelist.txt | xargs -n 1 -P 8 -I {} xrdcp root://cms-xrd-global.cern.ch/{} .


newName="DataRun3"

max_index=$(ls DataRun3_*.root 2>/dev/null | \
            sed -E 's/DataRun3_([0-9]+)\.root/\1/' | \
            sort -n | tail -1)

if [[ -z "$max_index" ]]; then
    next_index=1
else
    next_index=$((max_index + 1))
fi

for f in *.root; do
    if [[ "$f" =~ ^[0-9a-fA-F-]{36}\.root$ ]]; then
        new_name="${newName}_${next_index}.root"

        echo "$f → $new_name"
        mv "$f" "$new_name"

        next_index=$((next_index + 1))
    fi
done