def make_hooked_optimizer(optimizer_class, hook, *optimizer_args, **optimizer_kwargs):
    class HookedOptimizer(optimizer_class):
        def step(self, closure=None):
            super().step(closure)
            hook()

    return HookedOptimizer(*optimizer_args, **optimizer_kwargs)
