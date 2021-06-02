This project uses computer vision concepts to perform lane detection and yaw rate reporting
on road car videos.

Lane Detection preview:
https://youtube.com/embed/AwJPlQezIqQ
Here, I use a sliding window search over the entire frame after thresholding it to a black-and-white image
in order to identify the lane lines in the image.

Yaw Rate reporting preview:
https://youtube.com/embed/wTH6MbjkhVY
Here, I use OpenCV feature detection and feature matching in order to determine the general delta of the frames, and then I compare this delta to the vector of going straight ahead. The angle between these two vectors is the car's current yaw.

As you can see in the yaw rate reporting, the green arrow changes along with the direction of car travel. For example, when turning, the yaw arrow will move in the turning direction.

