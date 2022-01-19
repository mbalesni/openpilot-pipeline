import cv2 
import os 

"""
Algorithm:

1. Create 4 boxes on the image boundaries.
2. Do that for two consective frames 
3. match the points between two consecutve frames
4. Subtract the obtained boxes 
5. If the values are greater than some threshold for n number of frames for time span x
6. Return if the camera is stationary

"""

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        frames.append(frame)
        if not ret:
            break
    return frames

def return_boxes(img):
    
    top_box = img[:200,:,:]
    left_box = img[:,:200,:]
    bottom_box = img[-200:,:,:]
    right_box = img[:,-200:,:] 
    
    return top_box, left_box, bottom_box, right_box

def subtract_boxes(img1, img2 ):

    boxes_img1 = return_boxes(img1)
    boxes_img2 = return_boxes(img2)

    top = boxes_img1[0] - boxes_img2[0]
    left = boxes_img1[1] - boxes_img2[1]
    bottom = boxes_img1[2] - boxes_img2[2]
    right = boxes_img1[3] - boxes_img2[3]
    
    return top, left, bottom, right

def detect_moving_camera(path_to_video):

    frames = load_video(path_to_video)
    n_frames = 500
    count_threshold = 100000
    count_check = 0
    
    for i in range(n_frames):
        image1 =  frames[i]
        image2 = frames[i+1]

        t_diff, l_diff, b_diff, r_diff = subtract_boxes(image1, image2)

        if ((t_diff>0).sum()  and (l_diff>0).sum() and (b_diff>0).sum() and (r_diff>0).sum()) > count_threshold :
            flag = True

            if flag == True:
                count_check += 1
        elif ((t_diff>0).sum()  and (l_diff>0).sum() and (b_diff>0).sum() and (r_diff>0).sum()) < count_threshold:
            flag = False

            if flag ==False:
                count_check = 0 

        if count_check == 200 : ## check for consective first 200 frames 
            return True
            
if __name__ == "__main__":
    video_path = "/gpfs/space/projects/Bolt/comma_recordings/realdata/2021-09-07--08-22-59/14/fcamera.hevc"
    a = detect_moving_camera(video_path)
    if a == True:
        print("yes the car is moving")
    