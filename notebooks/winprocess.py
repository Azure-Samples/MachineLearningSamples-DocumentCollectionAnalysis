import inspect
import types


def execute_source(callback_imports, callback_name, callback_source, args):
    for callback_import in callback_imports:
        exec(callback_import, globals())
    exec('import time' + "\n" + callback_source)
    callback = locals()[callback_name]
    return callback(*args)


def submit(executor, callback, *args):
    callback_source = inspect.getsource(callback)
    callback_imports = list(imports(callback.__globals__))
    callback_name = callback.__name__
    future = executor.submit(
        execute_source,
        callback_imports, callback_name, callback_source, args
    )
    return future


def imports(callback_globals):
    for name, val in list(callback_globals.items()):
        if isinstance(val, types.ModuleType) and val.__name__ != 'builtins' and val.__name__ != __name__:
            import_line = 'import ' + val.__name__
            if val.__name__ != name:
                import_line += ' as ' + name
            yield import_line