# myCamera_Alvium.py - Python script containing functions to control the Alvium camera.
#
# GENERAL DESCRIPTION:
# ----------------------------------------------------------------------------------------------------------------------
# Functions to control the Alvium camera.
#
# INPUT: None
#
# OUTPUT: None
#
# ----------------------------------------------------------------------------------------------------------------------
#The MIT License (MIT)
#
#Copyright (c) 2023 Pablo Yubero
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
#  $Version: 1.0 $  $Date: 2023/02/15 $
#
# ----------------------------------------------------------------------------------------------------------------------

import cv2
import numbers
import numpy as np
from vimba import Vimba, PixelFormat
from datetime import datetime
from time import sleep
from threading import Thread

class myCamera:
    def __init__(self, camera_id = None, backend = cv2.CAP_DSHOW , color=False):
        
        #...define initial variables
        self.camera_id = camera_id
        self.backend   = backend
        self.vcapture  = cv2.VideoCapture() # cap object is created but it is not assigned to any camera 
        
        self._verbose      = True
        self._is_open      = False
        self._frame        = []
        self._exposure_auto = False

        self.thread_streaming_running = False
        self.thread_recording_running = False
        self.preview_running          = False

        self.recording_filename = 'video_NVIDEO.avi'
        self.recording_format   = 'MJPG'
        self.recording_fps      = 1
        self.recording_totaltime= 99999 #about 2.7h
        self.recording_maxtime  = 5
        self.recording_nvideo   = 0
        
        if color:
            self._pixelformat = PixelFormat.Bgr8
            print('pixel format set to bgr8')
        else:    
            self._pixelformat  = PixelFormat.Mono8
            print('pixel format set to mono8')
            
        
        # INITIALIZATION #
        if camera_id is not None:
            self.open( self.camera_id )
            # self.__update_properties() #called from within open()
            


        
    def __del__(self):
        self.close()


    def close(self):
        if self.thread_recording_running:
            self.stop_recording()
            
        if self.preview_running:
            self.stop_preview()
            
        if self.thread_streaming_running:
            self.stop_streaming()
                
        
        
    def open(self, camera_id = None):
        if camera_id is None:
            camera_id = self.camera_id
        else:
            self.camera_id = camera_id
            
        try:
            if self.thread_streaming_running:
                print('<< E >> Camera %d is already open.' % camera_id)
                return 
            
            with Vimba.get_instance() as vimba:
                cams = vimba.get_all_cameras()
                with cams[self.camera_id] as cam:
                    self.__update_properties(cam)
                    
            self._is_open = True
            print('Alvium camera %d can now be opened.' % camera_id )
            return True
        
        except:
            self._is_open = False
            print('<< E >> Failed to connect to Alvium camera %d.' % camera_id )
            return False
        
        
        
    def snapshot(self, formfactor = 1):
        # Slow way of taking pictures. It takes about 1s to open the camera and take a picture.
        # If you need taking pictures faster, we recommend to start a stream, and grab
        # pictures directly from self.frame.
        
        # A camera needs to be available
        if not self._is_open:
            print('<< E >> Please, open the camera before taking any snapshot.')
            return
        
        # If a stream is already running, return current frame
        if self.thread_streaming_running:
            pass
        #... if not, access the camera and update current frame
        else:
            with Vimba.get_instance() as vimba:
                cams = vimba.get_all_cameras()
                with cams[self.camera_id] as cam:
                    self.__set_properties(cam)
                    self.__update_properties(cam)
                    frame_alvium = cam.get_frame()
                    frame_alvium.convert_pixel_format( self._pixelformat ) 
                    
                    self.frame = frame_alvium.as_opencv_image()
                    
        if formfactor != 1:    
            print('Snapshot taken and resized by %1.2f.' % formfactor ) if self._verbose else None
            return cv2.resize( self.frame , None , fx=formfactor , fy=formfactor)
        else:
            print('Snapshot taken.') if self._verbose else None
            return self.frame


    #################################
    ####### MODIFY PROPERTIES #######
    # To prevent the camera from raising errors from mismanipulations,
    # we create a copy of the supposed properties of the camera as a dict.
    # Whenever we use set() we JUST change the value in the dictionnary.
    # Then whenever the camera is opened (snapshot and streaming), we try to upload
    # ...the values from the dict to the camera. 
    # The true property values of the camera are downloaded whenever the camera 
    # ... is instantiated and after uploading our own values from the dict.
    
    def get(self, propertyName):
        if propertyName in self.properties:
            return self.properties[propertyName]
        else:
            print('<< W >> %s could not be found in myCamera.properties.' % propertyName )
            return False

    def set(self, propertyName, value):
        if propertyName in self.properties:
            self.properties.update( {propertyName : value})
            print('Changed %s to %s' % (propertyName, str(value)) )
        else:
            print('<< W >> %s could not be found in myCamera.properties.' % propertyName )
            return False


    def __set_properties(self, cam):
        ''' This function is called when opening the camera in snapshot() and streaming().
        ExposureAuto is set to Once so that it adjust itselfs on initialization.
            You can then regulate lighting with the manual aperture.
        Gain is set to 0 to avoid any white-pixel noise. '''
        # if self.thread_streaming_running:
        #     print('<< E >> ')
        cam.Width.set( self.properties['width'] )
        cam.Height.set( self.properties['height'] )
        cam.Gain.set( 0.0 ) 
        
        # check exposure time
        val = cam.ExposureTime.get()
        inc = cam.ExposureTime.get_increment()
        delta = int( ( self.properties['exposure'] - val )/inc )
        cam.ExposureTime.set( val + delta*inc )
        
        #............
        #cam.ExposureTime.set( self.properties['exposure'] )
        # cam.GainAuto.set('Once')# self.properties['autogain'] )
        
    def __update_properties(self, cam):
        self.properties = {}
        self.properties.update( {'width'    : cam.Width.get() } )
        self.properties.update( {'height'   : cam.Height.get() } )
        self.properties.update( {'autoexposure' : cam.ExposureAuto.get() } )
        self.properties.update( {'exposure' : cam.ExposureTime.get() } )
        self.properties.update( {'autogain' : cam.GainAuto.get() } )
        self.properties.update( {'gain'     : cam.Gain.get()    })
        self.properties.update( {'temperature': cam.DeviceTemperature.get() })
        #self.properties.update( {'pxformat' : cam.PixelFormat.get() } )
                
    
    def __auto_exposure_fun(self, cam):
        print('Computing exposure time')
        cam.ExposureAuto.set('Once')
        # value = cam.ExposureTime.get()
        self._exposure_auto = False
    
    def set_auto_exposure(self):
        self._exposure_auto = True
        
        
    def summary(self):
        print('----- Summary -----')
        for key, value in self.properties.items():
            if isinstance( value, numbers.Number) :
                print("%12s - %1.1f" % (key, value) )
        
    #################################
    ######## STREAMING THREAD #######
    def start_streaming(self):
        
        if not self._is_open:
            print("<< E >> Please open a camera before starting the stream.")
            return False
        
        if self.thread_streaming_running and self.thread_streaming.is_alive():
            print("<< W >> A stream is already running.")
            return True
            
        self.thread_streaming = Thread(target = self.__streaming_fun, daemon = True) 
        self.thread_streaming.start()
        sleep(1.0)
        
        return True
    
    
    def __streaming_fun(self):
    
        self.streaming_myfps    = myFPS( averaging = 30 )
    
        with Vimba.get_instance() as vimba:
            cams = vimba.get_all_cameras()
            
            with cams[self.camera_id ] as cam:
                self.__set_properties(cam)
                self.__update_properties(cam)
                cam.start_streaming( self.__streaming_frame_handler )
                
                self.thread_streaming_running = True
                print('Streaming started.') if self._verbose else None
                
                while self.thread_streaming_running:
                    # Check whether to recalculate exposure time automatically...
                    if self._exposure_auto:
                        self.__auto_exposure_fun(cam)
                    
                    # Sleep some time so the max fps are 100
                    sleep( 0.01 ) 
                    
                self.__update_properties(cam)
                cam.stop_streaming()
    
    
    
    
    def __streaming_frame_handler(self, cam, frame_alvium):
        frame_alvium.convert_pixel_format(  PixelFormat.Bgr8 ) # I think this is unnecessary

        self.frame = frame_alvium.as_opencv_image()
        self.streaming_myfps.tick()  
        
        #... check overheating
        self.properties.update( {'temperature' : cam.DeviceTemperature.get() })
        self.properties.update( {'exposure' : cam.ExposureTime.get() } )
        if self.properties['temperature']>80:
            print('')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('Temperature over limit.')
            print('Aborting...')
            self.close()
        
        # This is important to continue the stream
        cam.queue_frame(frame_alvium)
 

    
    def stop_streaming(self):
        if self.thread_streaming_running:
            self.thread_streaming_running = False
            self.thread_streaming.join()
            print('Streaming stopped.') if self._verbose else None        

        else:
            print('<< W >> Streaming not found, it could not be stopped.')
        
        return True
    
      
    def streaming_fps(self):
        return self.streaming_myfps.get()
    




    #################################
    ########## PREVIEW EASY #########
    def start_preview(self, formfactor=1):
        #... start stream if there is none
        if self.start_streaming():
            print('Starting camera preview.')
            print('Press R to start recording with default filename.')
            print('Press Z to zoom in a rectangle.')
            print('Press Q to exit.')
            
            
            self.preview_running = True
            self.preview_zoom_bbox = None
            self.preview_ffactor_main = formfactor
            self.preview_ffactor_zoom = 1
            self.preview_winname_main = 'Preview. Press Q to exit.'
            self.preview_winname_zoom = 'Zoom region. Press Z to delete.'
            
            
            while self.preview_running:
                frame    = self._resize( self.frame, self.preview_ffactor_main)
                print(frame.shape)
                if len(frame.shape)==2 or frame.shape[2]==1:
                    frame    = cv2.cvtColor( frame, cv2.COLOR_GRAY2BGR )
                fps_text = '%1.1f +/- %1.1f' % self.streaming_fps() 
                temp_text= 'Temp = %2.1f' % self.properties['temperature']
                color    = (0,0,255)
                font     = cv2.FONT_HERSHEY_SIMPLEX
                frame    = cv2.putText(frame, fps_text,  (20,20), font, 0.5, color, 1, cv2.LINE_AA )
                frame    = cv2.putText(frame, temp_text, (20,40), font, 0.5, color, 1, cv2.LINE_AA )
                h,w,_ = frame.shape
                cv2.line(frame, 
                            ( int(0.45*w), int(h/2) ) ,
                            ( int(0.55*w), int(h/2) ),
                            color, 1) 
                cv2.line(frame, 
                            ( int(w/2), int(0.45*h) ) ,
                            ( int(w/2), int(0.55*h) ),
                            color, 1) 
                # Draw some text with information on the main frame
                if self.thread_recording_running:
                    rec_text = '[REC %ds]' % self.recording_time
                    cv2.putText(frame, rec_text, (20,60), font, 0.5, color, 20, cv2.LINE_AA )
                   
                    
                    
                # Create zoom window
                if self.preview_zoom_bbox:
                    x0,y0,w,h = self.preview_zoom_bbox
                    ff        = self.preview_ffactor_main
                    
                    #... crop original frame and display
                    zoom_frame = self.frame[int(y0):int(y0+h), int(x0):int(x0+w)]
                    zoom_frame = self._resize( zoom_frame, self.preview_ffactor_zoom)

                    cv2.imshow( self.preview_winname_zoom , zoom_frame )
                    cv2.setMouseCallback(self.preview_winname_zoom, lambda event,x,y,flags,params : self.__preview_callback(event,x,y,flags,'zoom') )
                    
                    #... draw on the main frame the zoomed rectangle
                    frame = cv2.rectangle( frame, ( int(x0*ff) ,int(y0*ff) ), ( int(x0*ff+w*ff), int(y0*ff+h*ff) ), (255,0,0), 1 )
                    

                cv2.imshow(self.preview_winname_main, frame  )
                cv2.setMouseCallback(self.preview_winname_main, lambda event,x,y,flags,params : self.__preview_callback(event,x,y,flags,'main') )
                
           
                key = cv2.waitKey(5)
                
                if key==ord('q'):
                    self.stop_preview()
                    
                elif key == ord('r'):
                    self.toggle_recording()
                    
                elif key == ord('e'):
                    self.set_auto_exposure()
                    
                elif key == ord('z'):
                    if self.preview_zoom_bbox:
                        self.preview_zoom_bbox=None
                        cv2.destroyAllWindows()
                    else:
                        cv2.destroyAllWindows()
                        self.preview_zoom_bbox = cv2.selectROI(frame, showCrosshair=False)
                        self.preview_zoom_bbox = [ value/self.preview_ffactor_main for value in self.preview_zoom_bbox]
                        cv2.destroyAllWindows()  
                    
            cv2.destroyAllWindows()
            self.preview_running = False
           
            
     
    def __preview_callback(self, event,x,y,flags,param):
        # Event on Left button double click
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if param =='main':
                self.preview_ffactor_main = 1.2*self.preview_ffactor_main
            if param =='zoom':
                self.preview_ffactor_zoom = 1.2*self.preview_ffactor_zoom
        
        # Event on Right button double click
        if event == cv2.EVENT_RBUTTONDBLCLK:
            if param =='main':
                self.preview_ffactor_main = 0.833*self.preview_ffactor_main
            if param =='zoom':
                self.preview_ffactor_zoom = 0.833*self.preview_ffactor_zoom
           
                
    def stop_preview(self):
        self.preview_running = False
        sleep(0.5)
        
    
    
    #################################
    ########## RECORDINGS ###########
    def start_recording(self, filename=None, fmt=None, total_time=None, fps=None ):
        if not self._is_open:
            print("<< E >> Please open a camera before recording any video.")
            return False
        
        if not self.start_streaming():
            print("<< E >> Could not start a video stream.")
            return False
        
        if self.thread_recording_running:
            print('<< E >> A recording is already running, please stop it first.')
            return False
         
        if filename   is None: filename   = self.recording_filename
        if fmt        is None: fmt        = self.recording_format
        if total_time is None: total_time = self.recording_totaltime
        if fps        is None: fps        = self.recording_fps
        
        self.recording_time    = 0
        self.recording_nframes = 0
        self.recording_start   = datetime.now()

        filename = filename.replace( 'NVIDEO', '%03d' % self.recording_nvideo ).replace('DATETIME', datetime.now().strftime('%y%m%d%H%M') )
        print('Recording... ' , filename )
        THREADFUN = lambda: self.__recording_fun(filename, fmt, total_time, fps)

        self.thread_recording = Thread(target = THREADFUN, daemon = True) 
        self.thread_recording.start()
        sleep(0.5)
                
        
    def stop_recording(self):
        if self.thread_recording_running:
            self.thread_recording_running = False
            self.thread_recording.join()
        else:
            print('<< W >> Recording not found, it could not be stopped.')
        return True
        

    def toggle_recording(self):
        if self.thread_recording_running and self.thread_recording.is_alive():
            self.stop_recording()
        else:
            self.start_recording()
            
            
    def __recording_fun(self, filename, fmt, total_time, fps ):
        color      = int(0)
        resolution = ( self.frame.shape[1], self.frame.shape[0] )
        
        fourcc     = cv2.VideoWriter_fourcc( *fmt )
        speed      = 30
        autorestart= False
        started_chunk = datetime.now()
        
        
        timer       = myTimer( 1.0/fps)
        videoWriter = cv2.VideoWriter(filename, fourcc, speed,  resolution, color )
        
        self.recording_nvideo += 1
        self.thread_recording_running = True
        
        
        #... then record every 1/FPS seconds
        while self.recording_time <= total_time and self.thread_recording_running:           
            self.recording_time = (datetime.now() - self.recording_start).total_seconds()
            recording_chunk     = (datetime.now() - started_chunk ).total_seconds()
            
            if timer.isTime():
                videoWriter.write( self.frame )
                self.recording_nframes += 1
                
            if recording_chunk >= self.recording_maxtime:
                self.thread_recording_running = False
                autorestart = True
                
                
        # signal that the recording has finished
        self.thread_recording_running = False
    
        if autorestart:
            self.__restart_recording()
    
    
    
    def __restart_recording(self):
        filename = self.recording_filename.replace( 'NVIDEO', '%03d' % self.recording_nvideo )
        fmt        = self.recording_format
        total_time = self.recording_totaltime
        fps        = self.recording_fps
       
        THREADFUN = lambda: self.__recording_fun(filename, fmt, total_time, fps)

        self.thread_recording = Thread(target = THREADFUN, daemon = True) 
        self.thread_recording.start()
        sleep(0.5)
        
        
    #################################
    ##### CONVENIENCE FUNCTIONS ##### 
    def _resize(self, frame, formfactor):
        return cv2.resize( frame, None , fx=formfactor, fy=formfactor )
        





class myFPS:
    def __init__(self, averaging = 10):
        self.lastTick=datetime(2,1,1,1,1)            #this would be tick number n
        self.previousTick=datetime(1,1,1,1,1)       #this would be tick n-1
        self.historic = np.ones((averaging,))
        self._ii = 0
        
    def tick(self):
        # Update ticks
        self.previousTick=self.lastTick
        self.lastTick=datetime.now()
        
        # Update FPS historic
        self.historic[ self._ii ] = 1/( (self.lastTick-self.previousTick).total_seconds() + 1e-6)
        self._ii += 1
        if self._ii == len( self.historic ):
            self._ii = 0
        
    def get(self):
        return np.mean(self.historic), np.std(self.historic)



class myTimer:
    def __init__(self,DeltaTime):
        self.DeltaTime=DeltaTime
        self.previous=datetime.now()
        
    def isTime(self):
        tFromPrevious=(datetime.now()-self.previous).total_seconds()
        if tFromPrevious>self.DeltaTime:
            self.previous = datetime.now()
            return True
        else :
            return False   



def number_of_cameras():
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        NumCams = len(cams)
        
    if NumCams>0:
        print('Total of Alvium cameras found: '+str(NumCams)+'.')
        return NumCams
    else:
        print('No Alvium cameras were found.')
        return NumCams





if __name__=='__main__':
    from matplotlib import pyplot as plt
    
    FILENAME = 'video_alvium.avi'
    FORMAT   = 'MJPG'
    LENGTH   = 10
    FPS      = 10
    
    #... open camera
    cam = myCamera(0, color=True)
    cam.set('width', 2592)
    cam.set('height', 1944)
    cam.set('exposure', 200000)
    #... take snapshot and plot it
    frame = cam.snapshot()
    plt.imshow(frame)
    
    #... starting streaming, or not, as you wish
    cam.start_streaming()   
 
    #... start preview, before which you can start a recording or not
    # cam.start_recording( FILENAME, FORMAT, LENGTH, FPS)
    cam.start_preview( formfactor=0.5)

    # cam.set('width',2592/4)
    # sleep(1)
    # frame = cam.snapshot()
    # print(frame.shape)
    # print('------------')
    # for key, value in cam.properties.items():
    #     print(key,'\t', value)
    # # cam.snapshot()

    # for key, value in cam.properties.items():
    #     print(key,'\t', value)
    #... stop streaming, or not, close will stop it    
    # cam.stop_streaming()
    cam.close()
    

























