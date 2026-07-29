# Hare-and-Hounds Recovery Test

## Abstract

Stellar surface rotation is routinely inferred from the quasi-periodic brightness modulations that starspots imprint on photometric light curves, but the mapping from a light curve back to the underlying spot distribution is degenerate: rotation period, differential-rotation shear, spot lifetime, and active latitude all leave overlapping signatures. A study by Aigrain et al. (2015) conducted a blind "hare-and-hounds" exercise, releasing 1000 simulated 1000-day light curves built from a butterfly-pattern spot-emergence model with **known injected** equatorial periods (10–50 d), differential-rotation shears (δΩ/Ω<sub>eq</sub> ≈ 0.1–1), spot decay timescales (1–10x rotation periods), and active-latitude ranges (0°–80°). 

Most curves were injected into real Kepler quiet-star photometry to carry realistic instrumental and stellar noise. Across five blind teams, rotation periods were recovered well (within 10% for the majority of detections), but the injected differential rotation was essentially unrecoverable — the exercise found little correlation between injected and recovered shear. The goal of this project will be to test if the physically-motivated Gaussian process model using `spotgp` can accurately infer differential rotation and spot properties. This will be a useful test to see how well the model works on noisy data while having ground-truth answers to compare to.


![](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/mnras/450/3/10.1093_mnras_stv853/3/stv853fig1.jpeg?Expires=1787859355&Signature=Gyn3dcCNkFfmr0lwsw~IWwpffAn6iaidUKSFq-3dFgRRzuilIKbJlkFk38kd83Gng7X7UahGc-BPpu7AvZDz3KKBkP6UsSX6crZZ-QQbUtck1nOu-LYV~ngHq5SVC6ZYA6R7foGaXGjgfow95bqs0186clGn36uSitFJcG1bFA0gm4hZjj2R8KOVxIGkZz81-HaeQBLJhVcy2Di9ZT2Y9FfXdY1AyagzbGT1ZieQLJxtMtmPBUDa3kn8AuBHkYbeqby8-wFUNg9SY3gO2pMwa9I7xHyb0Lt83ZW7Z5V5mw9DlD2PHLXD9JNQLTAT4RpqE~FmSs43d8hwTbOAt96NUQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

## Literature

Aigrain et al. 2015 - [Testing the recovery of stellar rotation signals from Kepler light curves using a blind hare-and-hounds exercise](https://academic.oup.com/mnras/article/450/3/3211/1072143)



