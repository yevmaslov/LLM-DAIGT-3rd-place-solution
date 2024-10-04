from types import SimpleNamespace


def dictionary_to_namespace(data):
    if type(data) is list:
        return list(map(dictionary_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dictionary_to_namespace(value))
        return sns
    else:
        return data


def namespace_to_dictionary(data):
    dictionary = vars(data)
    for k, v in dictionary.items():
        if type(v) is SimpleNamespace:
            v = namespace_to_dictionary(v)
        dictionary[k] = v
    return dictionary

