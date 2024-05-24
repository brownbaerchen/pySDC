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
        if type(problem) == str:
            from pySDC.projects.GPU.configs import get_problem

            return PathFormatter.format_problem(get_problem(problem))
        elif hasattr(problem, '__name__'):
            return f'{problem.__name__}'
        else:
            return f'{type(problem).__name__}'

    @classmethod
    def complete_fname(cls, **kwargs):
        out = ''
        for key in cls.formatters.keys():
            if key in kwargs.keys():
                formatted = cls.get_formatter(key)(kwargs[key])
                out += f'{cls.get_delimiter(key, formatted)}{formatted}'
        return out[1:]

    @classmethod
    def get_delimiter(cls, key, formatted):
        if formatted != '':
            return cls.delimiters.get(key, "_")
        else:
            return ''

    @staticmethod
    def format_index(index):
        if index is None:
            return ''
        return PathFormatter.log_solution.format_index(index)

    @staticmethod
    def to_str(args):
        return str(args)

    @staticmethod
    def useGPU_to_str(useGPU):
        if useGPU:
            return 'GPU'
        else:
            return 'CPU'

    @staticmethod
    def bool_to_str(args):
        if args:
            return 'Y'
        else:
            return 'N'

    formatters = {
        'base_path': to_str,
        'name': to_str,
        'problem': format_problem,
        'num_procs': format_procs,
        'useGPU': useGPU_to_str,
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
