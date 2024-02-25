import msgpack
import datetime
import numpy as np

def encoder(obj):
    if isinstance(obj, datetime.datetime):
        return {'__encoded_from': 'datetime.datetime', 'json-compatible': obj.strftime("%Y%m%dT%H:%M:%S.%f")}

    elif isinstance(obj, np.ndarray):
        return {'__encoded_from': 'np.ndarray', 'json-compatible': obj.tolist()}
    
    else:
        return obj

def decoder(obj):

    if not '__encoded_from' in obj:
        return obj

    if obj['__encoded_from'] == 'datetime.datetime':
        return datetime.datetime.strptime(obj["json-compatible"], "%Y%m%dT%H:%M:%S.%f")

    elif obj['__encoded_from'] == 'np.ndarray':
        return np.array(obj["json-compatible"])

    else:
        raise ValueError('unknown type')


def save_to_msgpack(data, path, encoder=encoder):
    packed_data = msgpack.packb(data, default = encoder)
    with open(path, 'wb') as file:
        file.write(packed_data) # type: ignore

def load_from_msgpack(path, decoder=decoder):

    with open(path, 'rb') as file:
        byte_data = file.read()

    # Deserialize the MessagePack byte string back to a Python object
    return msgpack.unpackb(byte_data, object_hook=decoder)

