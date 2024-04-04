from pySDC.core.ConvergenceController import ConvergenceController
import pickle


class LogStats(ConvergenceController):

    def setup(self, controller, params, *args, **kwargs):
        params['control_order'] = 999
        if 'hook' not in params.keys():
            from pySDC.implementations.hooks.log_solution import LogToFile

            params['hook'] = LogToFile

        self.counter = params['hook'].counter

        return super().setup(controller, params, *args, **kwargs)

    def post_step_processing(self, controller, S, **kwargs):
        hook = self.params.hook
        if self.counter < hook.counter:
            path = f'{hook.path}/{self.params.file_name}_{hook.format_index(self.counter)}.pickle'
            stats = controller.return_stats()
            with open(path, 'wb') as file:
                pickle.dump(stats, file)
                self.log(f'Stored stats in {path!r}', S)
            self.counter += 1
