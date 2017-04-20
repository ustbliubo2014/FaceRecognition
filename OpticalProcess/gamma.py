

import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import sys
import pdb
from scipy.misc import imsave, imread


def get_file(): 
    
    # Gets the file path from the user
    # path = raw_input("Enter file Path: ")
    path = 'e:/kugou/mv.mkv'

    # Opens the video file and stores it in vid
    vid = cv2.VideoCapture(path)

    # Opens the video to be ready to use
    vid.open(path)
    
    # Lets the program load the video
    cv2.waitKey(1000)

    # if the file can't be opened it tells the user and exits the program
    if vid.isOpened() is False:
        print "File failed to open! ", path
        sys.exit(1)

    return vid

    
def get_avg_pixel(pic): 
    # this function returns the average value of all the pixels in the image
    # Gets a three dimensional array with RGB Values of each pixel
    pixel = list(pic.getdata())
    # Sets avg be a double, and gets the height and width of the frame size
    avg = 0.0
    width, height = pic.size
    # loops through the photo, checking the every 5th pixel, so as to speed up the process
    for y in range(0, height-1, 5):
        for x in range(0, width-1, 5):
            # Gets the RGB value of each pixel
            pixRGB = pic.getpixel((x, y))
            R,G,B = pixRGB
            # Adds the average of the pixels to the average of the entire frame to check for brightness
            avg += (sum([R,G,B]) / 3.0)
    # calculates the average of the whole photo this gives it a number 0-255 which is the brightness of the photo.
    avg = avg / ((width/5) * (height/5))
    # if the brightness is too bright, it wont need to brighten the photo
    if avg > 50:
        avg = 0
    return avg



def lighten_photo(org_image):
    # the average of the pixels is the brightness of the photo.
    # if pic_brightness is 0 then the photo won't get changed
    # but the program will still create a new image.
    pic_brightness = get_avg_pixel(org_image)

    # Gets all the pixels from org_image and stores it in pixels
    pixels = org_image.getdata()
    
    # Sets the extent of how much to brighten the photo and changes it to a decimal.
    extent = 0.0
    extent += (pic_brightness / 100)
  
    # creates a new image to preserve the original
    new_image = Image.new('RGB', org_image.size)
    new_image_list = []

    # Adds the extent of the brightness to the multiplier.
    brightness_multiplier = 1.0 + extent

    # goes through and adds the changed pixels into an array for the new image
    for pixel in pixels:
        # if you change all three values the same amount the color doesn't change but the brightness does.
        new_pixel = (int(pixel[0] * brightness_multiplier),
                     int(pixel[1] * brightness_multiplier),
                     int(pixel[2] * brightness_multiplier))
        # checks to make sure every pixel is still in rgb range
        for pixel in new_pixel:
            if pixel > 255:
                pixel = 255
            elif pixel < 0:
                pixel = 0
        # Once the pixel is an acceptable pixel, places it in an array to create the new image
        new_image_list.append(new_pixel)

    # uses the array of pixels to create the new image.
    new_image.putdata(new_image_list)
    return new_image


def replace_photo(vid, pic):
    # Converts the image back from RGB to BGR
    new_frame = np.array(pic)
    # Convert RGB to BGR 
    new_frame = new_frame[:, :, ::-1].copy()
    # Writes the frame to the video
    vid.write(new_frame)


def get_video_image(vid):
    path = 'C:\Users\liubo-it\Desktop/1467031818.56.png'
    data = imread(path)
    frame = Image.fromarray(np.roll(data, 1, axis=-1))
    new_frame = lighten_photo(frame)
    imsave('C:\Users\liubo-it\Desktop/1467031818.56_gamma.png', np.array(new_frame))

    # # Reads in the frame from the video and returns a boolean into the variable flag if it was read in correctly
    # flag, frame = vid.read()
    #
    # # Gets the current frame position
    # frame_pos = vid.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    # print type(frame)
    # if flag:
    #     # Converts the frame from BGR to RGB using numpy
    #     data = np.asarray(frame)
    #     frame = Image.fromarray(np.roll(data, 1, axis=-1))
    #     # if everything is good, then it lightens photo
    #     print np.array(frame).shape
    #     new_frame = lighten_photo(frame)
    #     print np.array(new_frame).shape
    #     replace_photo(vid, new_frame)
    # else:
    #     # If the frame could not be read, then it sets the video back a frame so it can retry the current image
    #     cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_pos - 1)
    #     # Lets the next frame load if it couldn't be read correctly the first time
    #     cv2.waitKey(100)
    # return frame_pos

#END DEF

def main():
    
    #Gets the video file
    video = get_file()

    #loops while it is still retrieving images
    while True:
        
        #Gets the frame and lightens the photo if need be
        #and returns the frame position 
        frame_pos = get_video_image(video)
        
        #if the video reaches the end of the video, then it stops the loop
        #and ends the program
        if frame_pos == video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            break
        
    #release the video so it can close
    cv2.VideoCapture.release(video)
#END DEF

#This just starts the code and sends it to it's own function.
# main()

get_video_image('')