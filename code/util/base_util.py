import logging
import time
from contextlib import contextmanager
import hashlib
import multiprocessing
import pickle
import os
import configparser
import os
import json


class TimeInfo():
    def __init__(self):
        self.t0 = time.time()
        self.tag_dict = {}

    def now(self):
        return time.time()

    def get_delay_t0(self):
        return time.time() - self.t0

    def get_t0(self):
        return self.t0

    def start(self, tag='default'):
        self.tag_dict[tag] = time.time()

    def get_delay(self, tag='default'):
        return time.time() - self.tag_dict[tag]

    def rest(self, tag='default'):
        self.tag_dict[tag] = time.time()

    def end(self, tag='default'):
        d = time.time() - self.tag_dict[tag]
        del self.tag_dict[tag]
        return d


# 上下文管理器，主要用于计时使用
@contextmanager
def timer(title=''):
    ti = TimeInfo()
    if title == '':
        title = str(ti.get_t0())
    get_logger().info("|{}| - begin".format(title))
    yield ti
    get_logger().info("|{}| - done in {:.4f}s".format(title, ti.get_delay_t0()))


def md5(s):
    return hashlib.md5(str(s).encode('utf-8')).hexdigest()


logger_dict = {}


# 打log使用
def get_logger(filename='STD_LOG.log', logger_name='STD_LOG', level=logging.DEBUG, file_level=logging.INFO,
               fstr='%(asctime)s - %(name)s - %(levelname)s - %(message)s           |-| %(filename)s-%(funcName)s-%(lineno)s'):
    global logger_dict
    key = md5('{}-{}-{}-{}-{}'.format(filename, logger_name, level, file_level, fstr))

    if key not in logger_dict:
        # 创建一个logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # 定义handler的输出格式
        formatter = logging.Formatter(fstr)

        if filename is not None:
            # 创建一个handler，用于写入日志文件
            fh = logging.FileHandler(filename)
            fh.setLevel(file_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        # 给logger添加handler
        logger.addHandler(ch)
        logger_dict[key] = logger

    return logger_dict[key]


def pickle_dump(o, f):
    with timer('write obj to {}'.format(f)):
        with open(f, mode='wb') as f:
            pickle.dump(o, f)


def pickle_load(f):
    with timer('load obj from {}'.format(f)):
        with open(f, mode='rb') as f:
            return pickle.load(f)


# 多进程工具
class Worker(multiprocessing.Process):
    def __init__(self, target, *args, **kwargs):
        multiprocessing.Process.__init__(self)
        self.action = target
        self.kwargs = kwargs
        self.args = args
        ids = ''
        for param in args:
            ids = ids.join(str(id(param)))
        for key in kwargs:
            ids = ids.join(str(id(kwargs[key])))
        self.worker_tag_md5 = md5(ids)

    def run(self):
        # 字符串第一个是当前的线程名称，第二个是当前的方法的名称
        worker_tag = '{}-{}-{}'.format(multiprocessing.current_process().name,
                                       str(self.action).split(' ')[1].split('.')[-1],
                                       self.worker_tag_md5[:8])
        with timer(worker_tag):
            result = self.action(*self.args, **self.kwargs)
            if result:
                with open(self.worker_tag_md5, mode='wb') as f:
                    pickle.dump(result, f)

    def join(self, timeout=None):
        super().join(timeout)
        to_return = None
        if os.path.exists(self.worker_tag_md5):
            with open(self.worker_tag_md5, mode='rb') as f:
                to_return = pickle.load(f)
            os.remove(self.worker_tag_md5)
        return to_return


def get_worker_lock():
    return multiprocessing.Lock()


# 简单的ini配置文件处理
class IniConfigUtil(object):
    __DEFAULT_SECTION = 'CONFIG'
    __DEFAULT_FILE = 'config.ini'

    def __init__(self, config_file_path=__DEFAULT_FILE):
        self.cp = configparser.ConfigParser()
        self.config_file_path = config_file_path
        if os.path.exists(config_file_path):
            self.cp.read(config_file_path, 'utf-8')

    def put(self, key, value, section=__DEFAULT_SECTION):
        if not self.cp.has_section(section):
            self.cp.add_section(section)
        self.cp.set(section, key, value)
        self.cp.write(open(self.config_file_path, mode='w', encoding='utf-8'))

    def get(self, key, section=__DEFAULT_SECTION):
        if self.cp.has_option(section, key):
            return self.cp.get(section, key)
        return None


# 复杂的json配置文件处理
class JsonConfigUtil(object):
    __DEFAULT_FILE = 'config.json'

    def __init__(self, config_file_path=__DEFAULT_FILE, write2file=True):
        self.config_file_path = config_file_path
        self.write2file = write2file
        if os.path.exists(config_file_path):
            with open(config_file_path, encoding='utf-8') as cf:
                self.data = json.load(cf)
        else:
            self.data = {}
        self.dw = DictWrapper(self.data)

    def get_json_object(self):
        return self.data

    def put(self, key, value):
        self.dw.put(key, value)
        if self.write2file:
            with open(self.config_file_path, mode='w', encoding='utf-8') as cf:
                json.dump(self.data, cf, indent=4)

    def get(self, key):
        return self.dw.get(key)


# 包装dict，可以通过.来寻找元素
class DictWrapper():
    def __init__(self, data):
        self.data = data

    def get_dict(self):
        return self.data

    def put(self, key, value):
        if value == self.get(key):
            return
        data = self.data
        if isinstance(key, str):
            ks = key.split('.')
            for i in range(len(ks) - 1):
                if isinstance(data, dict):
                    if ks[i] not in data:
                        data[ks[i]] = {}
                    data = data[ks[i]]
                else:
                    raise Exception('key: %s is not a dict' % ks[i - 1])
            data[ks[len(ks) - 1]] = value

    def get(self, key):
        data = self.data
        if isinstance(key, str):
            ks = key.split('.')
            for i in range(len(ks)):
                if ks[i] in data:
                    data = data[ks[i]]
                else:
                    return None
            return data
        return None
