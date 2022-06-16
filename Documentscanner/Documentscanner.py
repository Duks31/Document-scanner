import cv2
import numpy as np

kernel = np.ones((5,5))


widthImg = 640
heightImg = 480
framewidth = 640
frameheight = 480

capture = cv2.VideoCapture(0) # capture webcam 

capture.set(3, framewidth)
capture.set(4, frameheight)
capture.set(10, 150)

# Some basic image preprocesssing 
def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray
    imgBlur = cv2.GaussianBlur(imgGray,(5,5), 1) # convert to blur
    imgCanny = cv2.Canny(imgBlur, 200, 200) # Image Edges
    imgdilate = cv2.dilate(imgCanny, kernel, iterations=1) # Images edges increased
    imgerode = cv2.erode(imgdilate, kernel, iterations=1)  #Images edges Reduced

    return imgerode



def get_contours(img):
    biggest = np.array([])
    maxArea = 0 
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
            if  area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area 

    cv2.drawContours(imgcontour, cnt, -1, (255,0,0), 5)
    return biggest 

# Getting points correctly 
def reorder(mypoints):
    mypoints = mypoints.reshape((4,2))
    mypointsnew = np.zeros((4,1,2), np.int32)
    add = mypoints.sum(1)   
    mypoints[0] = mypoints[np.argmin(add)]
    mypoints[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints, axis = 1)
    mypointsnew[1] = mypoints[np.argmin(diff)]   
    mypointsnew[2] = mypoints[np.argmax(diff)]   
    print('New Points', mypoints)
    return mypointsnew 

# Getting image in the correct view of the Document 
def getwrap(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[widthImg, 0],[heightImg,0],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgoutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # cropping image
    imgcropped = imgoutput[20:imgoutput.shape[0]-20, 20:imgoutput.shape[1]-20]  
    imgcropped = cv2.resize(imgcropped, (widthImg, heightImg))  
    return imgoutput

#while loop to display result after iterating through all frames 
while True:
    success, img = capture.read() 
    # img = cv2.resize(img(widthImg, heightImg))
    imgcontour = img.copy()
    imgthreshold = preProcessing(img)
    biggest = get_contours(imgthreshold)

    # if biggest(array.size) > 0:
    imgwrapped = getwrap(img,biggest)
    # else:   
    #     break 
    cv2.imshow('Result', imgthreshold)
    if cv2.waitKey (1) & 0xFF == ord('q'):
        break

# remove all windows 
capture.release()
cv2.destroyAllWindows()


