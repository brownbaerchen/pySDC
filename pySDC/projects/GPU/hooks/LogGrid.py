from pySDC.core.hooks import Hooks
import pickle


class LogGrid(Hooks):
    file_logger = None
    file_name = 'grid'

    @classmethod
    def get_path(cls):
        return f'{cls.file_logger.path}/{cls.file_name}.pickle'

    @classmethod
    def load(cls):
        with open(cls.get_path(), 'rb') as file:
            return pickle.load(file)

    def pre_run(self, step, level_number):
        import numpy as np

        self.file_logger()

        prob = step.levels[level_number].prob
        grid = prob.get_grid()

        with open(self.get_path(), 'wb') as file:
            try:
                pickle.dump(np.array([me.get() for me in grid]), file)
            except AttributeError:
                pickle.dump(grid, file)
