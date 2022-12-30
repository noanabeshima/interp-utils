from typing import List, Callable


def register_hook(module, hook_fn: Callable):
    if not hasattr(module, "cache"):
        module.cache = {}
    if not hasattr(module, "hooks"):
        module.hooks = {}
    if hook_fn.__name__ in module.hooks:
        print(
            "Warning: hook_fn already registered:",
            module.__class__.__name__,
            hook_fn.__name__,
        )
        return
    removable_handle = module.register_forward_hook(
        lambda module, input, output: hook_fn(module, input, output)
    )
    module.hooks[hook_fn.__name__] = removable_handle
    return removable_handle


def register_hooks(module, hook_fns=List[Callable]):
    removable_handles = []
    for hook_fn in hook_fns:
        removable_handles.append(register_hook(module, hook_fn))


def clear_cache(module):
    if hasattr(module, "cache"):
        for v in module.cache.values():
            del v
        module.cache = {}
    for child in module.children():
        clear_cache(child)


def remove_hooks(module, quiet=False):
    if hasattr(module, "hooks"):
        for hook_name, removable_handle in module.hooks.items():
            if quiet is False:
                print("Removing hook: ", module.__class__.__name__, hook_name)
            removable_handle.remove()
        module.hooks = {}
    clear_cache(module)
    for child in module.children():
        remove_hooks(child, quiet=quiet)
