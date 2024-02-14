from pySDC.implementations.datatype_classes.mesh import mesh
import numpy as np


class MultiComponentMesh(mesh):
    components = []

    def __new__(cls, init, *args, **kwargs):
        if isinstance(init, tuple):
            obj = super().__new__(cls, ((len(cls.components), *init[0]), *init[1:]), *args, **kwargs)
        else:
            obj = super().__new__(cls, init, *args, **kwargs)

        for comp, i in zip(cls.components, range(len(cls.components))):
            obj.__dict__[comp] = obj[i]
        return obj

    def __array_ufunc__(self, *args, **kwargs):
        results = super().__array_ufunc__(*args, **kwargs).view(type(self))

        if type(self) == type(results) and self.flags['OWNDATA']:
            for comp, i in zip(self.components, range(len(self.components))):
                results.__dict__[comp] = results[i]

        return results
