import functools
import torch

def oom_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args):
        oom=False
        try:
            func(*args)
        except Exception as e:
            if "CUDA out of memory" in str(e):
                oom=True
                torch.cuda.empty_cache()
                # The first two arguments are the model (an nn.Module) and the input (a torch.Tensor).
                # Calling model.cpu() moves the model and its parameters to the CPU in-place,
                # while input.cpu() returns a new tensor on the CPU, leaving the original input unchanged.
                args[0].cpu()
                input_cpu=args[1].cpu()
                func(args[0],input_cpu,*args[2:])
                args[0].cuda()
            else:
                raise e
        return oom
    return wrapper