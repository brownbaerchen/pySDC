from pySDC.core.Hooks import hooks
import pickle


class LogGrid(hooks):
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
        self.file_logger()

        grid = step.levels[level_number].prob.X

        with open(self.get_path(), 'wb') as file:
            try:
                pickle.dump(grid.get(), file)
            except AttributeError:
                pickle.dump(grid, file)
