# July 32, 9:12pm

* Failure to train perhaps to due to not having enough CPU memory



## Sources
* [torch.distributed.elastic.multiprocessing.errors.ChildFailedError](https://discuss.huggingface.co/t/torch-distributed-elastic-multiprocessing-errors-childfailederror/28242)
* `export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO``
```
Hey guys, I’m glad to announce I solved the issue on my side.
As can be seen I use multiple GPUs, which have sufficient memory for the use case.
HOWEVER! My issue was due to not enough CPU memory. That’s why my runs crashed and without any trace of the reason.
Once I allocated enough cpu (on my case I increased it from 32GB to 96+ GB).

If the CPU allocation is constant and you can not allocated more, I’m sure you can try solutions as compressed models, deepspeed optimization levels and more.

Good luck to future readers.
```


### Root cause:
* Not having enough cpu memory, 


# Solutions:
* perhaps move everything into nvme or offload the parameters to the cpu using deepspeed

## Log
```
commune@r1n2a6000bittensor:~/Andromeda$ accelerate launch train.py
[2023-08-01 01:04:13,441] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
[2023-08-01 01:04:16,624] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:04:16,634] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:04:16,641] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:04:16,669] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:04:16,712] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:04:16,720] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 208581 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 208582 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 208583 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 208584 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 208586 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: -9) local_rank: 4 (pid: 208585) of binary: /usr/bin/python3.10
Traceback (most recent call last):
  File "/home/commune/.local/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/accelerate_cli.py", line 45, in main
    args.func(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 964, in launch_command
    deepspeed_launcher(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 687, in deepspeed_launcher
    distrib_run.run(args)
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
=======================================================
train.py FAILED
-------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
-------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-08-01_01:06:47
  host      : r1n2a6000bittensor
  rank      : 4 (local_rank: 4)
  exitcode  : -9 (pid: 208585)
  error_file: <N/A>
  traceback : Signal 9 (SIGKILL) received by PID 208585
=======================================================
commune@r1n2a6000bittensor:~/Andromeda$ export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO

commune@r1n2a6000bittensor:~/Andromeda$ accelerate launch train.py
[2023-08-01 01:09:31,113] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
[I socket.cpp:566] [c10d] The server socket has started to listen on [::]:29500.
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:46392.
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:46406.
[2023-08-01 01:09:34,414] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:09:34,417] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:09:34,477] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
[2023-08-01 01:09:34,541] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
[2023-08-01 01:09:34,614] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:09:34,642] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209014 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209015 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209016 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209018 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209019 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: -9) local_rank: 3 (pid: 209017) of binary: /usr/bin/python3.10
Traceback (most recent call last):
  File "/home/commune/.local/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/accelerate_cli.py", line 45, in main
    args.func(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 964, in launch_command
    deepspeed_launcher(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 687, in deepspeed_launcher
    distrib_run.run(args)
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
=======================================================
train.py FAILED
-------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
-------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-08-01_01:11:46
  host      : r1n2a6000bittensor
  rank      : 3 (local_rank: 3)
  exitcode  : -9 (pid: 209017)
  error_file: <N/A>
  traceback : Signal 9 (SIGKILL) received by PID 209017
=======================================================
commune@r1n2a6000bittensor:~/Andromeda$ 
```
------
----

# Log2
* I reconfigurd the setting to utilize torch dynamo and offload parameters to nvme

```
 commune@r1n2a6000bittensor:~/Andromeda$ accelerate config
[2023-08-01 01:15:17,803] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
----------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine                                                                                              
----------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                      
multi-GPU                                                                                                 
How many different machines will you use (use more than 1 for multi-node training)? [1]:                  
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes                                        
----------------------------------------------------------------------------------------------------------Which dynamo backend would you like to use?                                                               
nvfuser                                                                                                   
Do you want to customize the defaults sent to torch.compile? [yes/NO]:                                    
Do you want to use DeepSpeed? [yes/NO]: yes                                                               
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: no                                    
----------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?                                                  
3                                                                                                         
----------------------------------------------------------------------------------------------------------Where to offload optimizer states?                                                                        
nvme                                                                                                      
----------------------------------------------------------------------------------------------------------Where to offload parameters?                                                                              
nvme                                                                                                      
Nvme Path to offload parameters?                                                                          
Nvme Path to offload optimizer states?                                                                    
How many gradient accumulation steps you're passing in your script? [1]:                                  
Do you want to use gradient clipping? [yes/NO]: yes
What is the gradient clipping value? [1.0]: 
Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: yes
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes
How many GPU(s) should be used for distributed training? [1]:6
----------------------------------------------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
fp8                                                                                                       
accelerate configuration saved at /home/commune/.cache/huggingface/accelerate/default_config.yaml         
commune@r1n2a6000bittensor:~/Andromeda$ accelerate launch train.py                   
[2023-08-01 01:15:58,494] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)                                                                                             
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
[I socket.cpp:566] [c10d] The server socket has started to listen on [::]:29500.
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:45830.
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:45838.
[2023-08-01 01:16:01,364] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:16:01,455] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:16:01,456] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
[2023-08-01 01:16:01,484] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:16:01,555] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
[2023-08-01 01:16:01,593] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209602 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209603 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209604 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209605 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 209606 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: -9) local_rank: 0 (pid: 209601) of binary: /usr/bin/python3.10
Traceback (most recent call last):
  File "/home/commune/.local/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/accelerate_cli.py", line 45, in main
    args.func(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 964, in launch_command
    deepspeed_launcher(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 687, in deepspeed_launcher
    distrib_run.run(args)
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
=======================================================
train.py FAILED
-------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
-------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-08-01_01:18:29
  host      : r1n2a6000bittensor
  rank      : 0 (local_rank: 0)
  exitcode  : -9 (pid: 209601)
  error_file: <N/A>
  traceback : Signal 9 (SIGKILL) received by PID 209601
=======================================================
```


# Log3
* I changed the config to use deepspeed1, same error

```
commune@r1n2a6000bittensor:~/Andromeda$ accelerate config
[2023-08-01 01:21:26,715] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
-----------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine                                                                                                                       
-----------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                               
multi-GPU                                                                                                                          
How many different machines will you use (use more than 1 for multi-node training)? [1]:                                           
Do you wish to optimize your script with torch dynamo?[yes/NO]:no                                                                  
Do you want to use DeepSpeed? [yes/NO]: yes                                                                                        
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: no                                                             
-----------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?                                                                           
1                                                                                                                                  
How many gradient accumulation steps you're passing in your script? [1]:                                                           
Do you want to use gradient clipping? [yes/NO]: no                                                                                 
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes                 
How many GPU(s) should be used for distributed training? [1]:6                                                                     
-----------------------------------------------------------------------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
fp8                                                                                                                                
accelerate configuration saved at /home/commune/.cache/huggingface/accelerate/default_config.yaml                                  
commune@r1n2a6000bittensor:~/Andromeda$ accelerate launch train.py                                                        
[2023-08-01 01:21:50,336] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)            
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
[I socket.cpp:566] [c10d] The server socket has started to listen on [::]:29500.
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:57524.
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:57530.
[2023-08-01 01:21:53,173] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:21:53,189] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:21:53,237] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
[2023-08-01 01:21:53,367] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:21:53,439] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-08-01 01:21:53,452] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 210195 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 210197 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 210198 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 210199 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 210200 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: -9) local_rank: 1 (pid: 210196) of binary: /usr/bin/python3.10
Traceback (most recent call last):
  File "/home/commune/.local/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/accelerate_cli.py", line 45, in main
    args.func(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 964, in launch_command
    deepspeed_launcher(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 687, in deepspeed_launcher
    distrib_run.run(args)
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
=======================================================
train.py FAILED
-------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
-------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-08-01_01:24:23
  host      : r1n2a6000bittensor
  rank      : 1 (local_rank: 1)
  exitcode  : -9 (pid: 210196)
  error_file: <N/A>
  traceback : Signal 9 (SIGKILL) received by PID 210196
=======================================================
commune@r1n2a6000bittensor:~/Andromeda$ 

```

# Log3
* No deepspeed at all but rather fullyshardeddataparallel with shardgradop,transformerbasedwrap,
sharded_state_dict,


```
ommune@r1n2a6000bittensor:~/Andromeda$ accelerate config
[2023-08-01 01:25:09,849] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
-----------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine                                                                                                                       
-----------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                               
multi-GPU                                                                                                                          
How many different machines will you use (use more than 1 for multi-node training)? [1]:                                           
Do you wish to optimize your script with torch dynamo?[yes/NO]:                                                                    
Do you want to use DeepSpeed? [yes/NO]:                                                                                            
Do you want to use FullyShardedDataParallel? [yes/NO]: yes                                                                         
-----------------------------------------------------------------------------------------------------------------------------------What should be your sharding strategy?                                                                                             
SHARD_GRAD_OP                                                                                                                      
Do you want to offload parameters and gradients to CPU? [yes/NO]: yes                                                              
-----------------------------------------------------------------------------------------------------------------------------------What should be your auto wrap policy?                                                                                              
TRANSFORMER_BASED_WRAP                                                                                                             
Specify the comma-separated list of transformer layer class names (case-sensitive) to wrap ,e.g, :`BertLayer`, `GPTJBlock`, `T5Block`, `BertLayer,BertEmbeddings,BertSelfOutput` ...? :                                                                               
-----------------------------------------------------------------------------------------------------------------------------------What should be your FSDP's backward prefetch policy?
BACKWARD_PRE                                                                                                                       
-----------------------------------------------------------------------------------------------------------------------------------What should be your FSDP's state dict type?                                                                                        
SHARDED_STATE_DICT                                                                                                                 
Do you want to enable FSDP's forward prefetch policy? [yes/NO]: yes                                                                
Do you want to enable FSDP's `use_orig_params` feature? [yes/NO]: yes                                                              
Do you want each individually wrapped FSDP unit to broadcast module parameters from rank 0 at the start? [yes/NO]:                 
How many GPU(s) should be used for distributed training? [1]:
-----------------------------------------------------------------------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
fp8                                                                                                                                
accelerate configuration saved at /home/commune/.cache/huggingface/accelerate/default_config.yaml                                  
commune@r1n2a6000bittensor:~/Andromeda$ accelerate launch train.py                                                                 
[2023-08-01 01:25:47,200] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)            
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
[I socket.cpp:566] [c10d] The server socket has started to listen on [::]:29500.
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:47910.
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:47916.
[2023-08-01 01:25:49,991] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/usr/lib/python3/dist-packages/requests/__init__.py:87: RequestsDependencyWarning: urllib3 (2.0.4) or chardet (4.0.0) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:45082.
[I socket.cpp:787] [c10d] The client socket has connected to [localhost]:29500 on [localhost]:45084.
[I ProcessGroupNCCL.cpp:665] [Rank 0] ProcessGroupNCCL initialized with following options:
NCCL_ASYNC_ERROR_HANDLING: 1
NCCL_DESYNC_DEBUG: 0
NCCL_BLOCKING_WAIT: 0
TIMEOUT(ms): 1800000
USE_HIGH_PRIORITY_STREAM: 0
[I ProcessGroupNCCL.cpp:842] [Rank 0] NCCL watchdog thread started!
Traceback (most recent call last):
  File "/home/commune/Andromeda/train.py", line 705, in <module>
    main()
  File "/home/commune/Andromeda/train.py", line 702, in main
    Train()
  File "/home/commune/Andromeda/train.py", line 484, in Train
    state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = CFG.BATCH_SIZE #??????
AttributeError: 'NoneType' object has no attribute 'deepspeed_config'
[I ProcessGroupNCCL.cpp:844] [Rank 0] NCCL watchdog thread terminated normally
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 210780) of binary: /usr/bin/python3.10
Traceback (most recent call last):
  File "/home/commune/.local/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/accelerate_cli.py", line 45, in main
    args.func(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 966, in launch_command
    multi_gpu_launcher(args)
  File "/home/commune/.local/lib/python3.10/site-packages/accelerate/commands/launch.py", line 646, in multi_gpu_launcher
    distrib_run.run(args)
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/commune/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-08-01_01:29:53
  host      : r1n2a6000bittensor
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 210780)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
commune@r1n2a6000bittensor:~/Andromeda$ 

```