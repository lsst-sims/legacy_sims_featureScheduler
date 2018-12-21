

The feature-based scheduler uses a modified Markov Decision Process to select where the best spot on the sky to observe would be. In these sub directories, we run a few simplified surveys to illustrate how they effect the decision process.

There are three basis functions that we tend to always use.

1. The `M5_diff_basis_function` combines the current skybrightness, seeing, and airmass all-sky maps to compute the 5-sigma depth at every point in the sky. It then computes the difference between the current 5-sigma map and an all-sky map that has the expected "best" 5-sigma depth a healpixel would reach. This can be thought of as computing how large a penalty one incurrs by observing any particular point in the sky. This basis function helps ensure reasonable filter choices are made, the penalty for observing in bright time is much lower in red filters than in blue filters.

2. The `Target_map_basis_function` tracks the progress of the survey and compares it to the goal map. This basis function rewards areas that have been under-observed and penalizes those that have been over-observed.

3. The `Slewtime_basis_function` rewards short slews and penalizes long slews. 

Other basis functions used in these examples include `Strict_filter_basis_function` that rewards staying in the same filter (to prevent excessive filter changing), and some functions that mask the zenith and region around the moon.

The final reward map is a linear combination of the above basis functions. The free parameters of the scheduler are the weights given to each basis function. The weights can be thought of as converting the different units of our observing desires to a common ground. The weights allow us to speciy that X magnitudes of lost depth is equal to observing a part of the sky that is Y observations behind. 


## Single Filter
----------------

The notebook xxx-path runs a serries of simulated surveys where the weight on each basis function is increased

| | Baseline  | Slewtime | m5  | Target Map  | 
|-- | :--------:  | :-------: | :----: | :------: |
| N obs| 24,611 |  25,012 | 24,010 | 23,106  |
|Alt-az | <img src="./1filter/default/thumb.default_Count_observationStartMJD_HEAL_SkyMap.png", width=150> | 
|RA,Dec |  <img src="./1filter/default/thumb.default_Count_observationStartMJD_r_HEAL_SkyMap.png", width=150>| 