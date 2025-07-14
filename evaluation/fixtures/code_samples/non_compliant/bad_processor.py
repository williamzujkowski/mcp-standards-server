# bad code with no docs
def process(d):
    r = []
    for i in d:
        try:
            r.append({'id': i['id'], 'done': 1})
        except:
            pass  # ignore errors
    return r

class processor:
    def __init__(self, c):
        self.c = c  # no validation

    def run(self, data):
        global result  # global variable
        result = []
        for x in data:
            result += [x]
        return result
