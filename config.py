import configparser
import os
import inspect


def read_conf(relative_file_path='/config.conf'):
    configuration = configparser.ConfigParser()
    conf_file_path = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    conf_file_path += relative_file_path
    read_conf_files = configuration.read(conf_file_path)

    if not read_conf_files:
        raise Exception('Config. file %s not found.', conf_file_path)

    return configuration['DEFAULT']

custome_config = None


class CustomConfig(object):
    data = read_conf()

    @classmethod
    def get(cls, name):
        if custome_config:
            return custome_config[name.lower()]
        return cls.data[name]

    @classmethod
    def getint(cls, name):
        if custome_config:
            return int(custome_config[name.lower()])
        return int(cls.data[name])

    @classmethod
    def getfloat(cls, name):
        if custome_config:
            return float(custome_config[name.lower()])
        return float(cls.data[name])

    @classmethod
    def getboolean(cls, name):
        if custome_config:
            return custome_config[name.lower()] == 'True'
        return cls.data[name] == 'True'

    @classmethod
    def change_config_file(cls, path):
        cls.data = read_conf(path)


def set_custome_config(data):
    global custome_config
    custome_config = data

config = CustomConfig
