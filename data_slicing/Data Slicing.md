# Data Slicing and Barrel Containment

<!-- #region -->
When trained on data where events originated in the center of the tank and hit only the barrel of the tank (constant polar angle of 0), the model performed better than when trained on the dataset where the position and direction of particles were varied.

There are 3 datasets that had been used at this time:
1. **Original barrel only dataset** - particles originate in center of tank and hit only in the barrel, `/data/WatChMaL/data/9M_IWCD_data_angles.h5`
2. **Center dataset** - particles originate in center tank but have varying polar angle, `/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_centre.h5`
3. **Varying position/angle dataset** - particle has varying starting position and polar angle, `/data/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_splits_CNN/IWCDmPMT_4pi_fulltank_9M_test.h5` and `/fast_scratch/WatChMaL/data/IWCDmPMT_4pi_fulltank_9M_splits_CNN/IWCDmPMT_4pi_fulltank_9M_trainval.h5`


To determine the difference in performance caused by varying the polar angle and the location of the particle in the tank, a script was used to slice events within a certain distance of the center of the tank and where the hits were contained within the barrel (not hitting the end caps).

The notebook `Varying Position Dataset` contains the function `find_bounds` used to perform the slicing for the dataset (for this particular notebook, that is the varying position dataset). This function takes as input the position, angle, label, and energy arrays for the events and returns the max and min angle at which cherenkov radiation will hit the tank, as well as the max and min distance of the Cherenkov ring to the wall of the barrel.

This notebook was also used to generate indices for the contained events. These indices were stored in the files `varpos_trainval_indices.npz` and `varpos_test_indices.npz` for the training/validation and test sets, respectively.

Other notebooks performed the same purpose for other datasets, but they were accidentally deleted during a github commit - but subbing out the dataset pathway for another dataset should easily allow the same slicing and indexing to be performed for a different dataset.

This notebook also displays event images for the sliced data in two separate formats. The first shows individual PMTs within mPMTS, and the second displays each mPMT as one point in the image. `PlottingEventDisplays.py` contains the functions used for making these displays.
<!-- #endregion -->

```bash

```
