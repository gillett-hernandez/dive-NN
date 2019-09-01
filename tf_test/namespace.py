class Namespace(dict):
    """Namespace. subclass of ``dict``. allows access of keys using dot syntax.

:Example:

>>> ns = Namespace({})
>>> ns.stuff = 1
>>> ns.stuff2 = 2
>>> print(ns.stuff * ns.stuff2)
2
>>> print(ns.stuff * ns["stuff2"])
2
>>> existing_nested_dictionary = {"foo": {"bar": "baz", "spam": ["eggs1", "eggs2"]}, "etc": {}}
>>> ns_from_dict = Namespace.transform(existing_nested_dictionary)
>>> ns_from_dict.foo
Namespace({"bar": "baz", "spam":["eggs1", "eggs2"]})
>>> ns_from_dict.etc
Namespace({})
>>> assert ns_from_dict.foo.bar == "baz"
>>> assert ns_from_dict.get("lol", False) is False
"""

    def __getattr__(self, key):
        try:
            return super().__getitem__(key)
        except AttributeError:
            print(key)
            raise
        except KeyError:
            print(key)
            raise AttributeError(f"couldn't find key {key}")

    def __setattr__(self, key, value):
        super().__setitem__(key, value)

    @classmethod
    def transform(cls, blob):
        if isinstance(blob, dict):
            blob = blob.copy()
            for k, v in blob.items():
                blob[k] = cls.transform(blob[k])
            return cls(blob)
        elif isinstance(blob, list):
            return [cls.transform(e) for e in blob]
        else:
            return blob

    def __repr__(self):
        return "Namespace(" + str({key: value for key, value in self.items()}) + ")"
