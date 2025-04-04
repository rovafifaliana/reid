import numpy as np
import cv2

def extractOpticalFlow(img1, img2):
    totalPoints = [[],[]]
    try:
        # params for corner detection 
        feature_params = dict( maxCorners = 100, 
                            qualityLevel = 0.5, 
                            minDistance = 10, 
                            blockSize = 10)

        # Parameters for lucas kanade optical flow 
        lk_params = dict( winSize = (15, 15), 
                        maxLevel = 2, 
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                                    10, 0.03)) 
        # Create some random colors 
        #color = np.random.randint(0, 255, (100, 3)) 

        # Take first frame and find corners in it 

        img1_gray = cv2.cvtColor(img1, 
                                cv2.COLOR_BGR2GRAY) 
        p0 = cv2.goodFeaturesToTrack(img1_gray, mask = None, 
                                    **feature_params) 
        # Create a mask image for drawing purposes 
        #mask = np.zeros_like(img1) 

        img2_gray = cv2.cvtColor(img2, 
                                    cv2.COLOR_BGR2GRAY) 
        # calculate optical flow 
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, 
                                                img2_gray, 
                                                p0, None, 
                                                **lk_params) 
        # Select good points 
        good_new = p1[st == 1] 
        good_old = p0[st == 1] 

        # draw the tracks 
        for (new, old) in zip(good_new,good_old): 
            a, b = new.ravel() 
            c, d = old.ravel()
            totalPoints[0].append(int(a-c))
            totalPoints[1].append(int(b-d))
        
    except Exception as exception:
        print("ExtractOpticalFlow: ",exception)

    return np.median(totalPoints[0]),np.median(totalPoints[1])
