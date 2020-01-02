import os
import sys
import json
import requests
import numpy as np

class Timeit:
    '''
    Measure elapsed time of a block of code
    Usage:
    with Timeit('my code block'):
        # code here
        # code here
        # code here
    
    output:
    Code block  'my code block' took 67.0860 ms
    '''
    def __init__(self, name=None):
        self.name = "'"  + name + "'" if name else ''

    def __enter__(self):
        self.start = time.clock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (time.clock() - self.start) * 1000.0
        print('Code block %s took %.4f ms' % (self.name, self.took))

def join_path(*args):
    return os.path.join(*args)


def slack_message(message, webhook_file="/home/burak/.local/slack_wh.txt"):
    """Sends notification to Slack webhook.
    Webook address should be defined in webhook_file as a single line.

    Arguments:
        message {[type]} -- [description]
    """
    with open(webhook_file) as f:
        url = f.readline().strip()
    if isinstance(message, str):
        message = json.dumps({"text": message})
    elif not isinstance(message, dict):
        print("Invalid type as Slack message.")

    requests.post(url=url, data=message)


def test():
    slack_message("Test general message.")


if __name__ == "__main__":
    test()