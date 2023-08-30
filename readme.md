# IsoLearner
IsoLearner is a deep learning model designed to predict relative isotopologue abundances from bulk metabolomics data, as outlined in 'Predicting Isotopologue Counts From Bulk Unlabeled Metabolomics Data' (Bisram 2023). The model is trained on a large dataset of stable-isotope tracing experiments using MALDI-MSI and is able to accurately predict the proportions of isotopologue counts in the absence of labeled tracers. By leveraging the power of deep learning, our model is able to capture complex relationships between metabolites and predict isotope tracing data with state-of-the-art accuracy.

## Sample Usage
        # Instantiate the IsoLeaner model, loading in the dataset and cleaning out bad metabolites/isotopologues
        Brain_3HB_IsoLearner = IsoLearner.IsoLearner(absolute_data_path = "/Users/bisramr/MATLAB/Projects/Isoscope_Matlab_V/generated-data",
                                            relative_data_path = "brain-m0-no-log/Brain-3HB")

        Brain_3HB_IsoLearner.cross_validation_training()

        ground_truth, predictions = Brain_3HB_IsoLearner.cross_validation_testing()

        Brain_3HB_IsoLearner.cross_validation_eval_metrics(ground_truth, predictions)

        vis.cross_validation_results(ground_truth, predictions, coords_df = Brain_3HB_IsoLearner.coords_df, iso_to_plot = 'Anserine m+03', limited = False)

## Parameters:

## Methods: 

## Attributes: 
