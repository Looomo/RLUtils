from datetime import datetime,timedelta
import os
import time
     
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

    def flat_dict(self, _dict):
        data = []
        for k,v in _dict.items():
            if type(v) == float or type(v) == int:
                data.append( f"{k}: {v:.8f}" )
            else:
                data.append( f"{k}: {str(v)}" )
        info = f"\r\n\t".join( data )
        return info

    def debug(self, message,flush = True):
        self.log(  message, flush=flush , level = self.DEBUG  )
    
    def info(self, message,flush = True):
        self.log(  message, flush=flush , level = self.INFO  )

    def warning(self, message,flush = True):
        self.log(  message, flush=flush , level = self.WARNING  )

    def error(self, message,flush = True):
        self.log(  message, flush=flush , level = self.ERROR  )

    def log(self, info, level = 1, flush = True, ):

        if type(info) is dict: info = self.flat_dict(info)

        if level < self.output_level:
            return
        current_time = datetime.now()
        info_with_time = f"[{current_time}] [ {self.level_strs[level]} ] {info}"
        cmd = f"echo \"{info_with_time}\" >> {self.logfile}"
        os.system(cmd)
        print(info_with_time, flush=flush)
        return


class TimedMyLogger(MyLogger):
    def __init__(self, logfile="log.log") -> None:
        super().__init__(logfile)
        self._start = time.time()
        self._last_report = 0

    def time_diff(self, reset = True ):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff
    
    def log(self, info, level = 1, flush = True, progress_info: dict = None ):
        
        
        # {"step":0, "total": 100}
        if progress_info is not None:

            time_span = self.time_diff()
            step_span = progress_info["step"] - self._last_report
            self._last_report = progress_info["step"]
            time_remain = time_span*(progress_info["total"] - progress_info["step"])/step_span
            eta = str(timedelta(seconds=time_remain)) 


            progress = 100*progress_info["step"]/progress_info["total"]

            time_info = f"========== {progress_info['step']}/{progress_info['total']} [ {progress:.2f}% ] ========="
            eta_time_info = f"========== ETA: {eta}   Timecost Since Last Report: {time_span} secs ========="

            super().log(info=time_info, level=level, flush=flush)
            super().log(info=info, level=level, flush=flush)
            super().log(info=eta_time_info, level=level, flush=flush)

        else:
            super().log(info=info, level=level, flush=flush)



        # if level < self.output_level:
        #     return
        # current_time = datetime.now()
        # info_with_time = f"[{current_time}] [ {self.level_strs[level]} ] {info}"
        # cmd = f"echo \"{info_with_time}\" >> {self.logfile}"
        # os.system(cmd)
        # print(info_with_time, flush=flush)
        return