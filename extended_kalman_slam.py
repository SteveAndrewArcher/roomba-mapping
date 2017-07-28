

import math
import numpy as np
import pylab

class ExtendedKalmanFilterSLAM:
    def __init__(self, state, covariance, robot_width, control_motion_factor, control_turn_factor):
        
        self.state = state
        self.covariance = covariance

        self.robot_width = robot_width
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = 5.0  
        self.measurement_angle_stddev = math.radians(2)

        self.number_of_landmarks = 0
        self.numCorrections = 0
        
    
    @staticmethod
    def g(state, control, w):
        x, y, theta = state[0:3]
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            g1 = x + (rad + w/2.)*(math.sin(theta+alpha) - math.sin(theta))
            g2 = y + (rad + w/2.)*(-math.cos(theta+alpha) + math.cos(theta))
            g3 = (theta + alpha) % (2*math.pi)
        else:
            g1 = x + l * math.cos(theta)
            g2 = y + l * math.sin(theta)
            g3 = theta
        
        return np.array([g1, g2, g3])
    
    
    @staticmethod
    def h(state, landmark):
        dx = landmark[0] - state[0]
        dy = landmark[1] - state[1]
        r = math.sqrt(dx * dx + dy * dy)
        alpha = (math.atan2(dy, dx) - state[2]) % (2*math.pi)

        return np.array([r, alpha])
    
    @staticmethod
    def dg_dstate(state, control, width):
        theta = state[2]
        l,r = control
        if r != l:
            alpha = (r-l)/width
            radius = l/alpha
            dg1x = 1
            dg1y = 0
            dg1theta = (radius + width/2.)*(math.cos(theta+alpha)-math.cos(theta))
            dg2x = 0
            dg2y = 1
            dg2theta = (radius + width/2.)*(math.sin(theta+alpha)-math.sin(theta))
            dg3x = 0
            dg3y = 0
            dg3theta = 1
        else:
            dg1x = 1
            dg1y = 0
            dg1theta = -l*math.sin(theta)
            dg2x = 0
            dg2y = 1
            dg2theta = l*math.cos(theta)
            dg3x = 0
            dg3y = 0
            dg3theta = 1
        return np.array([[dg1x,dg1y,dg1theta],[dg2x,dg2y,dg2theta],[dg3x,dg3y,dg3theta]])
    
    @staticmethod
    def dg_dcontrol(state, control, w):
        theta = state[2]
        l, r = tuple(control)
        if r != l:
            rml = r - l
            rml2 = rml * rml
            theta_ = theta + rml/w
            dg1dl = w*r/rml2*(math.sin(theta_)-math.sin(theta)) - (r+l)/(2*rml)*math.cos(theta_)
            dg2dl = w*r/rml2*(-math.cos(theta_)+math.cos(theta)) - (r+l)/(2*rml)*math.sin(theta_)
            dg1dr = (-w*l)/rml2*(math.sin(theta_)-math.sin(theta)) + (r+l)/(2*rml)*math.cos(theta_)
            dg2dr = (-w*l)/rml2*(-math.cos(theta_)+math.cos(theta)) + (r+l)/(2*rml)*math.sin(theta_)
            
        else:
            dg1dl = 0.5*(math.cos(theta) + l/w*math.sin(theta))
            dg2dl = 0.5*(math.sin(theta) - l/w*math.cos(theta))
            dg1dr = 0.5*(-l/w*math.sin(theta) + math.cos(theta))
            dg2dr = 0.5*(l/w*math.cos(theta) + math.sin(theta))

        dg3dl = -1.0/w
        dg3dr = 1.0/w
        return np.array([[dg1dl, dg1dr], [dg2dl, dg2dr], [dg3dl, dg3dr]])
    
    @staticmethod
    def dh_dstate(state, landmark):
        dx = landmark[0] - state[0]
        dy = landmark[1] - state[1]
        q = dx**2 + dy**2
        rx = -dx/math.sqrt(q)
        ry = -dy/math.sqrt(q)
        rtheta = 0
        alphax = dy/q
        alphay =  -dx/q
        alphatheta = -1
        return np.array([[rx, ry, rtheta], [alphax, alphay, alphatheta]])
    
    @staticmethod
    def get_error_ellipse(covariance):
        
        eigenvals, eigenvects = np.linalg.eig(covariance[0:2,0:2])
        angle = math.atan2(eigenvects[1,0], eigenvects[0,0])
        result = angle, math.sqrt(eigenvals[0]), math.sqrt(eigenvals[1])
        return result
        
    
    def get_landmarks(self):

        return ([(self.state[3+2*j], self.state[3+2*j+1])
                 for j in xrange(self.number_of_landmarks)])

    def get_landmark_error_ellipses(self):
        
        ellipses = []
        for i in xrange(self.number_of_landmarks):
            j = 3 + 2 * i
            ellipses.append(self.get_error_ellipse(
                self.covariance[j:j+2, j:j+2]))
        return ellipses

    def predict(self, control):
        G3 = self.dg_dstate(self.state, control, self.robot_width)
        left, right = control
        left_var = (self.control_motion_factor * left)**2 +\
                   (self.control_turn_factor * (left-right))**2
        right_var = (self.control_motion_factor * right)**2 +\
                    (self.control_turn_factor * (left-right))**2
        control_covariance = np.diag([left_var, right_var])
        V = self.dg_dcontrol(self.state, control, self.robot_width)
        R3 = np.dot(V, np.dot(control_covariance, V.T))
        
        G = np.eye(3 + 2*self.number_of_landmarks)
        G[0:3,0:3] = G3

        R = np.zeros((3 + 2*self.number_of_landmarks,3 + 2*self.number_of_landmarks))
        R[0:3,0:3] = R3
         
        # covariance' = G * covariance * GT + R
        # where R = V * (covariance in control space) * VT.
        # Covariance in control space depends on move distance.
        self.covariance = np.dot(G, np.dot(self.covariance, G.T)) + R 
        # state' = g(state, control)
        self.state[0:3] = self.g(self.state, control, self.robot_width)
        
    def add_landmark_to_state(self, initial_coords):
    
        self.state = np.append(self.state, initial_coords)
        
        self.number_of_landmarks += 1
        
        newDimension = 3+2*self.number_of_landmarks
        
        newCovariance = np.zeros((newDimension,newDimension))
        newCovariance[0:newDimension-2,0:newDimension-2] = self.covariance
        newCovariance[newDimension-2:newDimension,newDimension-2:newDimension] = np.diag([1e10,1e10])   
        
        self.covariance = newCovariance
        return self.number_of_landmarks-1
    
    def correct(self, measurement, landmark_index):
        landmark = self.state[3+2*landmark_index : 3+2*landmark_index+2]
        H3 = self.dh_dstate(self.state, landmark)
        H = np.zeros((2,3+2*self.number_of_landmarks))
        H[0:2,0:3] = H3
        H[0:2,3+2*landmark_index:3+2*landmark_index+2] = np.negative(H3[0:2,0:2])
        Q = np.diag([self.measurement_distance_stddev**2,
                  self.measurement_angle_stddev**2])
        K = np.dot(self.covariance,
                np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(self.covariance, H.T)) + Q)))
        innovation = np.array(measurement) - self.h(self.state, landmark)
        innovation[1] = (innovation[1] + math.pi) % (2*math.pi) - math.pi
        if (self.state!=self.state + np.dot(K, innovation)).all():
            ekfs.numCorrections += 1
        self.state = self.state + np.dot(K, innovation)
        self.covariance = np.dot(np.eye(np.size(self.state)) - np.dot(K, H), self.covariance)
    
def getMovement(distanceTraveled,angleDegrees,w):
    wheelDifference = (angleDegrees*w*math.pi)/360.
    l = distanceTraveled - wheelDifference
    r = distanceTraveled + wheelDifference
    return np.array([l,r])

def getSensors(bumper, wall, ir, state, max_cylinder_distance):
    result = []
    if bumper == 3 or bumper == 2 or bumper == 1 :
        if bumper == 3:
            #front bumper
            angle = 0
        if bumper == 1:
            #right bumper
            angle = math.radians(-45)
        if bumper == 2:
            #left bumper
            angle = math.radians(45)
        distance = 170.
        result.append(bestLandmark((distance,angle), state, max_cylinder_distance))
    if wall == 1:
        angle = math.radians(-65)
        distance = 200.0
        result.append(bestLandmark((distance,angle), state, max_cylinder_distance))
    return (result)

def bestLandmark(measurement,state,max_cylinder_distance):
    distance,angle = measurement
    
    #position of landmark in relation to the robot
    xs, ys = distance*math.cos(angle), distance*math.sin(angle)
    
    #convert to position in world coordinates
    x,y = scanner_to_world(state[0:3],(xs,ys))
    
    best_dist_2 = max_cylinder_distance * max_cylinder_distance
    best_index = -1
    for index in xrange(((np.size(state)-3)/2)): 
        pole_x, pole_y = state[3+2*index : 3+2*index+2]
        dx, dy = pole_x - x, pole_y - y
        dist_2 = dx * dx + dy * dy
        if dist_2 < best_dist_2:
            best_dist_2 = dist_2
            best_index = index
    return(measurement, (x,y), (xs,ys), best_index)

def scanner_to_world(pose, point):
    dx = math.cos(pose[2])
    dy = math.sin(pose[2])
    x, y = point
    return (x * dx - y * dy + pose[0], x * dy + y * dx + pose[1])

def write_points(file_desc, line_header, cylinder_list):
    print >> file_desc, line_header,
    for c in cylinder_list:
        print >> file_desc, "%.1f %.1f" % c,
    print >> file_desc
    
def write_error_ellipses(file_desc, line_header, error_ellipse_list):
    print >> file_desc, line_header,
    for e in error_ellipse_list:
        print >> file_desc, "%.3f %.1f %.1f" % e,
    print >> file_desc

if __name__ == '__main__':

    
    start_state = np.array([2500.0, 2500.0, 90.0 / 180.0 * math.pi])
    start_covariance = np.zeros((3,3))
    bot_width = 242.
    motion_error_factor = .35
    turn_error_factor = .8
    max_cylinder_distance = 300.
    
    data =[]
    path = []
    landmarks = []
    totalTraveled = 0
    
    with open("roombadata.txt","r") as d:
        for packet in d:
            data.append(packet)
    messageswithsensors = 0
    ekfs = ExtendedKalmanFilterSLAM(start_state, start_covariance, bot_width, motion_error_factor, turn_error_factor)
    first = 1
    
    f = open("roombamap.txt", "w")
    landmarksadded = 0
    landmarksfound = 0
    
    print ekfs.state[2]
    for packet in data:
        
        bump,wall,distance,angle,ir = tuple([float(x) for x in packet.split(" ")])
        control = getMovement(distance,angle,bot_width)
        ekfs.predict(control)
        sensors = getSensors(bump,wall,ir,ekfs.state,max_cylinder_distance)
        for sensed in sensors:
            messageswithsensors += 1
            measurement, position, scannerposition, index = sensed
            if index == -1:
                #-1 is marker value for if the landmark should be added
                index = ekfs.add_landmark_to_state(position)
                landmarksadded += 1
            else:
                #otherwise just log that a previously found landmark is being considered
                landmarksfound += 1
                
            ekfs.correct(measurement,index)
        path.append([ekfs.state[0],ekfs.state[1]])
                
        #output robot's position
        print >> f, "F %f %f %f" % tuple(ekfs.state[0:3])
        # Write covariance matrix in angle stddev1 stddev2 stddev-heading form.
        e = ExtendedKalmanFilterSLAM.get_error_ellipse(ekfs.covariance)
        print >> f, "E %f %f %f %f" % (e + (math.sqrt(ekfs.covariance[2,2]),))
        # Write estimates of landmarks.
        write_points(f, "W C", ekfs.get_landmarks())
        # Write error ellipses of landmarks.
        write_error_ellipses(f, "W E", ekfs.get_landmark_error_ellipses())
        # Write cylinders detected by the scanner.
        write_points(f, "D C", [(scannerposition[0], scannerposition[1])
                                   for sensed in sensors])
    
    f.close()
    for i in range(3,np.size(ekfs.state),2):
        landmarks.append([ekfs.state[i],ekfs.state[i+1]])
    print len(landmarks)
    print np.size(ekfs.state)
    print "landmarks added ",landmarksadded
    print "landmarksfound ",landmarksfound
    print "corrections made ",ekfs.numCorrections
    print "messages with sensors", messageswithsensors 
    print "path 100", path[100]                 
    for point in path[::100]:
        pylab.plot([p[0] for p in path],[p[1] for p in path], 'bo')
    
    for point in landmarks:
        pylab.plot([p[0] for p in landmarks],[p[1] for p in landmarks], 'ro')
    pylab.ylim([0,5000])
    pylab.xlim([0,5000])
    pylab.show()
    