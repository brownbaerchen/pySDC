from pySDC.implementations.datatype_classes.mesh import mesh


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
