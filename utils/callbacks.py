# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Callback utils."""

import threading


class Callbacks:
    def __init__(self): #建立事件名单
        self._callbacks = {
            #字典：事件名称->这个事件需要执行的函数列表(里面也是字典,登记函数可以看出)
            "on_pretrain_routine_start": [],
            "on_pretrain_routine_end": [],
            "on_train_start": [],
            "on_train_epoch_start": [],
            "on_train_batch_start": [],
            "optimizer_step": [],
            "on_before_zero_grad": [],
            "on_train_batch_end": [],
            "on_train_epoch_end": [],
            "on_val_start": [],
            "on_val_batch_start": [],
            "on_val_image_end": [],
            "on_val_batch_end": [],
            "on_val_end": [],
            "on_fit_epoch_end": [],  
            "on_model_save": [],
            "on_train_end": [],
            "on_params_update": [],
            "teardown": [],
        }
        #停止训练标志 False表示继续训练，True表示停止训练，默认为False
        self.stop_training = False 


    """
    register_action是登记函数
    hook:事件名称 
    name:给这个函数起的名字 
    callback:真正要执行的函数
    """
    def register_action(self, hook, name="", callback=None):
        #检查传入的事件名称是否存在,否则输出错误
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        #检查callback是不是函数,否则输出错误  Tips: callable() 用来判断一个对象能不能像函数一样调用。
        assert callable(callback), f"callback '{callback}' is not callable"
        #把函数加入事件列表
        self._callbacks[hook].append({"name": name, "callback": callback})


    """
    run触发事件,负责执行已经登记过的函数
    callbacks.run("on_train_start")意思是on_train_start 事件发生了，请执行这个事件下的所有函数。
    """
    def run(self, hook, *args, thread=False, **kwargs):
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        #遍历这个事件的所有函数(登记的是字典)
        for logger in self._callbacks[hook]:
            if thread: #判断是否可以使用线程
                threading.Thread(target=logger["callback"], args=args, kwargs=kwargs, daemon=True).start()
            else:
                logger["callback"](*args, **kwargs)
