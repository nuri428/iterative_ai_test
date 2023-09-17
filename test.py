import dvc.api 
params = dvc.api.params_show(stages='train')
print(params)