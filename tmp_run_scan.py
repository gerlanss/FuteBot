from data.database import Database
from pipeline.scheduler import Scheduler

s = Scheduler(Database())
s._job_scanner()
