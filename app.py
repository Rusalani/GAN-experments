import cv2
import sys
from os import listdir
#imagePath = sys.argv[1]
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

imagePath = 'data/'
count=0
for f in listdir(imagePath):
    #print(imagePath+f)
    image = cv2.imread(imagePath+f,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30)
    )

    #print("[INFO] Found {0} Faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        finpic = image_resize(roi_color,512)
        status = cv2.imwrite('faces/'+str(count)+'.jpg', finpic)
        
        
        count+=1

print("done")
