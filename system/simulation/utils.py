import datetime


def get_time_str():
    return datetime.datetime.now().isoformat(timespec='minutes').replace(":", "_")
