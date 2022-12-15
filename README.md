# The BCRobo dataset for Robotic Vision and Autonomous Path Planning in Outdoor Beach Environment
***Author:*** Sam Tan Chi Jie (chijie1998@hotmail.com/ tan.jie-chi339@mail.kyutech.jp)

BCRobo dataset is a highly specialized dataset that contains high resolution beach environment images captured by a field exploration robot, SOMA.

[SOMA](https://alife-robotics.co.jp/members2020/icarob/data/html/data/OS/OS23/OS23-5.pdf).

As part of the development team for SOMA, we are trying to implement our robot in beach environment. However, we could not find any semantic dataset for beach environment and decided to make our own dataset. We made this public for everyone to use it.


![alt text](images/preview.jpg)


### Class and Labels
The labels of this dataset is adapted from [KITTI](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf) and [RUGD](http://rugd.vision/) dataset. 

There is a total of 22 classes but in the labels.txt file, we keep two extra classes (asphalt and picnic-table) as we wish to train it with RUGD dataset.


![alt text](images/pixel_percentage.png)


### Dataset Download

    .
    ├── ...
    ├── Jinoshima           # 393 images
    │   ├── ori          
    │   ├── anno        
    ├──  Agawa Hosenguri    # 292 images
    │   ├── ori          
    │   ├── anno        
    └── ...
    
The RGB and annotated ground truth images are available for download [here].

Lidar and GPS data are also available in the form of ROSbag upon request.

### Wiring with Arduino Uno or other microcontroller
Please refer to connection of Teensy 4.1, just connect the wires according to your microcontroller SPI pins. 

Please becareful of the voltage if you are using Arduino Uno, you need to step down the voltage on the SPI lines and input voltage to 3.3V as the IMU is running at 3.3V. 

### Kalman Filter
Modify Kalman Filter from [Osoyoo](https://github.com/osoyoo/Osoyoo-development-kits/tree/master/OSOYOO%202WD%20Balance%20Car%20Robot) to be compatible with this library to obtain tilted angle from IMU data. 

### Installation 
1. Download Arduino IDE and Teensyduino following this [official guide](https://www.pjrc.com/teensy/td_download.html). If you are not using teensy you may ignore this, you only need the arduino IDE. 

2. Find your Arduino folder usually located at Home. Download and put the ICM42688 and KalmanFilter folder in Arduino/libraries.

3. Open Arduino IDE, select Sketch and include libraries. You should be able to see and choose ICM42688 and KalmanFilter.

4. You may follow the guide from [Arduino](https://docs.arduino.cc/software/ide-v1/tutorials/installing-libraries) too if step 2 and 3 does not work for you.

### Available Functions 
![alt text](docs/images/function.jpg)

You may refer to the Example and readme on how to call and use the functions.

librarytest.ino demonstrates on how to use the basic functions to initialize and get gyro and accel data to compute tilted angle using Kalman Filter. 

IMUandKalman.ino demonstrated on how to use interrupt in Teensy to get IMU data every 50ms and print on serial monitor. 

### Credit and References

[ICM42688 datasheet](https://datasheet.octopart.com/ICM-42688-P-InvenSense-datasheet-140604332.pdf)

[Mikroe Libraries](https://www.mikroe.com/6dof-imu-14-click)

[ICM20948 Arduino Libraries](https://github.com/dtornqvist/icm-20948-arduino-library)
