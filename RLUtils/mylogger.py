from datetime import datetime
import os
class MyLogger():
    
    
    def __init__(self, logfile = "log.log") -> None:
        self.DEBUG = 0
        self.INFO = 1
        self.WARNING = 2
        self.ERROR = 3
        self.logfile = logfile
        self.output_level = 0
        self.level_strs = [
            'Debug',
            'Info',
            'Warning',
            'Error'
        ]
        # 0 for debug
        # 1 for info
        # 2 for warning
        # 3 for error

        self.info(  f"Logger initalized at {self.logfile} with logging level {self.output_level}."   )  
        pass
    def metainfo(self,update = False):
        if update:
            info = f"Logger initalized at {self.logfile} with logging level {self.output_level}."
        else:
            info = f"Updated: {self.logfile} with logging level {self.output_level}."
        self.info(  info   )  

    def set_logfile(self, logfile):
        self.logfile = logfile
        self.metainfo(update=True)
    
    def set_loglevel(self, level):
        self.output_level = level
        self.metainfo(update=True)

    def debug(self, message,flush = True):
        self.log(  message, flush=flush , level = self.DEBUG  )
    
    def info(self, message,flush = True):
        self.log(  message, flush=flush , level = self.INFO  )

    def warning(self, message,flush = True):
        self.log(  message, flush=flush , level = self.WARNING  )

    def error(self, message,flush = True):
        self.log(  message, flush=flush , level = self.ERROR  )

    def log(self, info, level = 1, flush = True, ):
        if level < self.output_level:
            return
        current_time = datetime.now()
        info_with_time = f"[{current_time}] [ {self.level_strs[level]} ] {info}"
        cmd = f"echo \"{info_with_time}\" >> {self.logfile}"
        os.system(cmd)
        print(info_with_time, flush=flush)
        return
