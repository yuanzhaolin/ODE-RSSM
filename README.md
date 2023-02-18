# ODE-RSSM


## How to train

The default configurations are in the `config.yaml` file.

Training the ODE-RSSM and RSSM with three datasets. When $D>1$, latent overshooting is used. 

```
python -m ode-rssm.model_train --multirun save_dir=final_rssm dataset=cstr,winding,southeast model=rssm ct_time=true sp=0.25,0.5 sp_even=false,true train.batch_size=2048 model.D=1,10
python -m ode-rssm.model_train --multirun save_dir=final_ode_rssm dataset=cstr,winding,southeast model=ode_rssm ct_time=true sp=0.25,0.5 sp_even=false,true train.batch_size=2048 model.D=1,10
```
The training logs, ckpt, and test results are saved in ```ckpt/${dataset.type}/${save_dir}/${model.type}_${parapeters}/${now:%Y-%m-%d_%H-%M-%S}```.
- ```figs```: A part of visualized predicted results whose number depends on the parameter ```test.plt_cnt```
- ```best.pth```: The model parameters with the best validation loss
- ```log.out```: Training logs
- ```train_loss.png```:  The training loss curve
- ```val_loss.png```:  The comparison of the validation loss and the training loss
- ```test.out```:  Test log


## Evaluating a trained model on test datset
python model_test.py test.plt_single=true test.test_dir\='{ckpt_path}'

Remember to add '/' before ```=``` and ```'```

An example of evaluating a trained model on test dataset.
```
python -m ode-rssm.model_test test.plt_single=true test.test_dir\=\'./ckpt/winding/ct_True/final_ode_rssm/ode_rssm_ct_time\=True,model.D\=10,random_seed\=0,sp\=0.25,sp_even\=False,train.batch_size\=2048/2022-05-15_17-18-00\'
```


