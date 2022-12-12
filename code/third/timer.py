import time

class Timer:
    
    start_time = 0
    end_time = 0
    elapsed_time = 0

    def __init__(self):
       pass

  
    def tic(self):
        self.start_time = time.time()

 
    def toc(self, string):
        if self.start_time:
            self.end_time = time.time()
            self.elapsed_time = self.end_time - self.start_time
            self.print_time(string)

            self.start_time = 0 #reset timer
        else:
            print("timer was not started, no time could be measured.")

    
        return self

    
    def print_time(self, string):

        if(self.elapsed_time < 0.1):
            print("---{0}: {1} ms.".format(string, self.elapsed_time * 1000))
        else:
            print("---{0}: {1} s.".format(string, self.elapsed_time))







