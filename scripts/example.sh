#!/usr/bin/env bash
cd ..

# Train
python ode-rssm.model_train --multirun save_dir=final_rssm dataset=cstr,winding,southeast model=rssm ct_time=true sp=0.25,0.5 sp_even=false,true train.batch_size=2048 model.D=1,10
python ode-rssm.model_train --multirun save_dir=final_ode_rssm dataset=cstr,winding,southeast model=ode_rssm ct_time=true sp=0.25,0.5 sp_even=false,true train.batch_size=2048 model.D=1,10

# Test

python -m ode-rssm.model_test test.plt_single=true test.test_dir\=\'./ckpt/winding/ct_True/final_ode_rssm/ode_rssm_ct_time\=True,model.D\=10,random_seed\=0,sp\=0.25,sp_even\=False,train.batch_size\=2048/2022-05-15_17-18-00\'

