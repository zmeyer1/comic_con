import rospy 
from geometry_msgs.msg import Twist

def move_quori():
    rospy.init_node('move_quori', anonymous=True)

    pub = rospy.Publisher('/quori/cmd_vel', Twist, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz

    move_cmd = Twist()
    move_cmd.linear.x = 0.5  
    move_cmd.angular.z = 0.0  # No rotation
    for i in range(20):
        pub.publish(move_cmd)
        rate.sleep()
    move_cmd.linear.x = 0.0  # Stop moving forward
    rospy.loginfo("Quori stopping ")
    for i in range(10):
        pub.publish(move_cmd)
        rate.sleep()






    move_cmd.linear.x = -0.5
    for i in range(20):
        pub.publish(move_cmd)
        rate.sleep()

    move_cmd.linear.x = 0.0  # Stop moving backward
    rospy.loginfo("Quori stopping ")
    pub.publish(move_cmd)

    if __name__ == '__main__':
        try:
            move_quori()
        except rospy.ROSInterruptException:
            pass