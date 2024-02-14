from pySDC.implementations.datatype_classes.mesh import mesh


class MultiComponentMesh(mesh):
    components = []

    def __new__(cls, init, *args, **kwargs):
        if isinstance(init, tuple):
            shape = (init[0],) if type(init[0]) is int else init[0]
            obj = super().__new__(cls, ((len(cls.components), *shape), *init[1:]), *args, **kwargs)
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

    # def __array_finalize__(self, obj):
    #     if obj is None:
    #         return
    #     super().__array_finalize__(self, obj)
    #     for comp, i in zip(self.components, range(len(self.components))):
    #         self.__dict__[comp] = obj.__dict__.get(comp, None)


class imex_mesh(MultiComponentMesh):
    components = ['impl', 'expl']


class comp2_mesh(MultiComponentMesh):
    components = ['comp1', 'comp2']
