class PathFormatter:
    log_solution = None

    @staticmethod
    def get_formatter(name):
        return PathFormatter.formatters[name]

    @staticmethod
    def format_procs(procs):
        if type(procs[0]).__name__ == 'Intracomm':
            return f'{procs[0].rank}_{procs[1].rank}_{procs[2].rank}'
        else:
            return f'{procs[0]-1}_{procs[1]-1}_{procs[2]-1}'

    @staticmethod
    def format_problem(problem):
        if hasattr(problem, '__name__'):
            return f'{problem.__name__}'
        else:
            return f'{type(problem).__name__}'

    @classmethod
    def complete_fname(cls, **kwargs):
        out = ''
        for key in cls.formatters.keys():
            if key in kwargs.keys():
                out += f'{cls.delimiters.get(key, "_")}{cls.get_formatter(key)(kwargs[key])}'
        return out[1:]

    @staticmethod
    def format_index(index):
        return PathFormatter.log_solution.format_index(index)

    @staticmethod
    def to_str(args):
        return str(args)

    formatters = {
        'base_path': to_str,
        'name': to_str,
        'problem': format_problem,
        'num_procs': format_procs,
        'space_resolution': to_str,
        'space_levels': to_str,
        'restart_idx': format_index,
        'format': to_str,
    }

    delimiters = {
        'format': '.',
        'name': '/',
    }


class DummyLogger(object):
    file_path = None
    path = None
    process_solution = None
    file_logger = None
