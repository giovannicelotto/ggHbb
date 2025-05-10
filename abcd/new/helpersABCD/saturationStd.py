import numpy as np
from scipy.stats import pearsonr

def saturationTest(bins,df, bootstrapEvents=[25,50,100], fractions=[0.25, 0.5, 0.75,]):
    
    
    stds={
        'bin_idx':[],
        'fraction':[],
        'nBoot':[],
        'std':[]
        }
    for bin_idx, (b_low, b_high) in enumerate(zip(bins[:-1], bins[1:])):
        print("Mass bin %d"%bin_idx, flush=True)

        m = (df.dijet_mass > b_low) & (df.dijet_mass < b_high)
        x = np.array(df.PNN1[m], dtype=np.float64)
        y = np.array(df.PNN2[m], dtype=np.float64)


        for fraction in fractions:
            print("Fraction %.2f"%fraction, flush=True)
            for nBoot in bootstrapEvents:
                print("nBoot %d"%nBoot, flush=True)
                boot_corrs = []
                for loopBoot_i in range(nBoot):
        # Compute Pearson correlation for the full bin

            # Bootstrap resampling
                    n_samples = int(len(x) * fraction)  
                    idx_data = np.random.choice(len(x), n_samples, replace=True)
                    boot_corrs.append(pearsonr(x[idx_data], y[idx_data])[0])
        
                std = np.std(boot_corrs)  
                stds['bin_idx'].append(bin_idx)
                stds['fraction'].append(fraction)
                stds['nBoot'].append(nBoot)
                stds['std'].append(std)

    return stds
