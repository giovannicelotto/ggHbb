cd /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
cmsenv

cd /t3home/gcelotto/ggHbb/WSFit/datacards/

# Define category groups
declare -A groups
groups[a]="0"
groups[b]="1 2 3 4"
groups[c]="1 2 6"
groups[d]="5 3 4"
groups[e]="5 6"
groups[f]="1 7 8"
groups[g]="10 20"
groups[h]="11 12 13 14 21 22 23 24"
groups[i]="11 12 16 21 22 23 24"
groups[j]="15 13 14 25 23 24"
groups[k]="15 16 25 26"
groups[l]="11 17 18 21 27 28"

outdir="combination_datacards"
mkdir -p $outdir
mkdir -p fitDiagnostics_expected

for letter in "${!groups[@]}"; do

    cats=${groups[$letter]}
    arr=($cats)

    if [ ${#arr[@]} -eq 1 ]; then
        # single category → just copy
        cp datacardMulti${arr[0]}.txt $outdir/datacardMulti${letter}.txt
    else
        # build combineCards command
        cmd="combineCards.py"
        for c in "${arr[@]}"; do
            cmd="$cmd cat${c}=datacardMulti${c}.txt"
        done

        $cmd > $outdir/datacardMulti${letter}.txt
    fi

    # Build freezeParameters string automatically
    freeze=""
    for c in "${arr[@]}"; do
        freeze="${freeze}pdfindex_${c}_2016_13TeV,"
    done
    freeze=${freeze%,}   # remove trailing comma

    echo "Running category group $letter (cats: $cats)"
    echo "Freezing: $freeze"

    combine -M FitDiagnostics \
        -d $outdir/datacardMulti${letter}.txt \
        -t -1 --expectSignal 1 \
        --X-rtd MINIMIZER_freezeDisassociatedParams \
        --setParameterRange r=-50,50:rateZbb=-3,5 \
        --freezeParameters $freeze \
        --cminDefaultMinimizerStrategy 0  \
        --robustFit 1 \
        > fitDiagnostics_expected/fitDiagnostics_expected_cat${letter}.txt

done