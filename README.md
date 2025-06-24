# Private repository for results on modelling intensity and possibility of teleworking

There are 3 different choice situations with 2 models for each choice situtation:
1. Intensity of teleworking with variables defined at the days of teleworking (e.g. an intensity of teleworking of 45% is counted as 3 days of teleworking) with an Ordinal Logit model and an ordinal RUMBoost model,
2. Intensity of teleworking with variables defined at the half days of teleworking (e.g. an intensity of teleworking of 45% is counted as 2.5 days of teleworking) with an Ordinal Logit model and an Ordinal RUMBoost model,
3. Possibility of teleworking with a binary logit model and a binary RUMBoost model.

For each choice situation and each model, the results are repeated 10 times to mitigate the randomness induced by the train-test split. Results are aggregated [here](https://github.com/NicoSlvd/simba-python-choice-private/blob/results-nicolas/src/simba/mobi/choice/data/output/results_analysis.ipynb).
Raw results are available [here](https://github.com/NicoSlvd/simba-python-choice-private/tree/results-nicolas/src/simba/mobi/choice/data/output/homeoffice/models/estimation/2021), 
with files "{metrics}/{parameteters_dcm}\_wfh\_{intensityX}/{possibility}\_seedY\_.csv" giving {metrics} values 
(Cross-entropy Loss for both choice situations and Mean Absolute Error, Mean Squared Error, Expected Mean Absolute Error 
and Expected Mean Squared Error only when it is an ordinal chocie situation) or {parameters} values for work from home intensity with days (X=20) or half days (X=10) or possibility and run Y.
RUMBoost model metrics follow the same logic with "rumboost_" in front of the file name. There are also raw RUMBoost models (files in .json extension) and raw biogeme results (files with .html and .pickle extensions, naming following the same logic)
