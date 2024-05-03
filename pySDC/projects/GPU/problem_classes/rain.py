import numpy as np

class Rain:
    xp = np
    rain_params = {}

    def set_random_generator(self, seed):
        import numpy as np
        self.rng = np.random.default_rng(seed=seed)

    def single_drop(self, x=None, y=None, sigma=None):
        x = self.rain_params.get('x', x)
        y = self.rain_params.get('y', y)
        sigma = self.rain_params.get('sigma', sigma)

        me = self.dtype_u(self.init, val=0.0)

        r2 = (self.X[0] - x) ** 2 + (self.X[1] - y) ** 2
        tmp = self.xp.exp(- r2 / sigma**2 / 2.)

        if self.spectral:
            me[:] = self.fft.forward(tmp)
        else:
            self.xp.copyto(me, tmp)

        return me

    def get_random_params(self, x=None, y=None, sigma=None):
        x = self.rain_params.get('x', x)
        y = self.rain_params.get('y', y)
        sigma = self.rain_params.get('sigma', sigma)

        self.rain_params = {
                'x': self.rng.uniform(self.x0, self.x0 + self.L[0]) if x is None else x,
                'y': self.rng.uniform(self.x0, self.x0 + self.L[1]) if y is None else y,
                'sigma': self.rng.uniform(self.L[0] / 100, self.L[0] / 10) if sigma is None else sigma,
        }
        return self.rain_params

    def random_drop(self, *args, **kwargs):
        return self.single_drop(**self.get_random_params(*args, **kwargs))
