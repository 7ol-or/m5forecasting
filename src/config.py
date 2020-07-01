import json
import os

class _flags(dict):
    def __init__(self, *args, **kwargs):
        super(_flags, self).__init__(*args, **kwargs)
        self._initialized = False

    def __getattr__(self, item):
        if item in self:
            return self.__getitem__(item)
        else:
            raise KeyError(item)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def _define(self, key, value):
        if key in self:
            print(f'{key} is already defined as {self[key]}')
        self.__setitem__(key, value)

    def _defines(self, dic):
        if hasattr(dic, '__dict__'):
            dic = dic.__dict__
        if not isinstance(dic, dict):
            raise TypeError(type(dic))
        for k, v in dic.items():
            self._define(k, v)

    def initialize(self, experiment):
        if not self._initialized:
            self['experiment'] = experiment
            profile_file = os.path.join(self['EXPERIMENTS_PATH'], experiment+'.json')
            with open(profile_file, 'rb') as fp:
                settings = json.load(fp)
            self._defines(settings)
            self._initialized = True

    @property
    def is_initialized(self):
        return self._initialized


FLAGS = _flags()
FLAGS._define('ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
FLAGS._define('DATA_PATH', os.path.join(FLAGS.ROOT, 'data'))
FLAGS._define('ARTIFACTS_PATH', os.path.join(FLAGS.ROOT, 'artifacts'))
FLAGS._define('EXPERIMENTS_PATH', os.path.join(FLAGS.ROOT, 'experiments'))
FLAGS._define('SUBMISSIONS_PATH', os.path.join(FLAGS.ROOT, 'submissions'))

for path in FLAGS:
    if path.endswith('PATH'):
        os.makedirs(FLAGS[path], exist_ok=True)