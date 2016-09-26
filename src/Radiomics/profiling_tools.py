import functools
import datetime

def time(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        start = datetime.datetime.today()
        result = func(*args, **kwargs)
        end = datetime.datetime.today()

        print "Elapsed time: ", end - start, "seconds"

        return result
    
    return wrapper