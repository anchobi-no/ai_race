import subprocess
import os
import time
import rospy


PREPARE_CMD = './prepare.sh -g False -l {}'
SEASONS = ['1s', '1w', '1f', '1c']
PREPARE_CWD = '/home/sakamoto/catkin_ws2/src/ai_race/scripts'
START_CAR_OPERATION_CMD = 'python car_operation.py -pt {} -eps {} -season {}'
STOP_CMD = '/home/sakamoto/catkin_ws2/src/ai_race/scripts/stop.sh'


class SeasonOperation:
    
    def __init__(self, time, pt=False):
        self.season = 0
        self.pt = pt
        self.count = 0
        self.start_train(time)


    def prepare_gazebo(self):
        cmd = PREPARE_CMD
        cmd = cmd.format(SEASONS[self.season])
        subprocess.Popen(cmd, cwd=PREPARE_CWD, shell=True)#, env=self.env)
        self.season += 1
        self.season %= 4
        time.sleep(12)
        

    def stop_gazebo(self):
        subprocess.call(STOP_CMD)
        time.sleep(20)


    def start_car_operation(self):
        cmd = START_CAR_OPERATION_CMD
        if not self.pt:
            self.pt = True
            pt = 'n'
        else:
            pt = 'y'
        if self.count < 4:
            eps = 0.2
        elif self.count < 8:
            eps = 0.1
        else:
            eps = 0
        cmd = cmd.format(pt, eps, SEASONS[self.season])
        subprocess.call(cmd, shell=True)


    def start_train(self, time):
        for i in range(time * 4):
            self.prepare_gazebo()
            self.start_car_operation()
            self.stop_gazebo()
            self.count += 1


def main():
    s_operation = SeasonOperation(20, False) 


if __name__ == '__main__':
    main()
