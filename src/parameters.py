class Parameters:
    def save_parameters(self, ignore=None, **kwargs):
        ignore = ignore or []
        for key, value in kwargs.items():
            if key not in ignore: setattr(self, key, value)